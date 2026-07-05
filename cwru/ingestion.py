import os
import scipy.io
import numpy as np
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def _get_temp_path(path: str, suffix: str = "tmp") -> str:
    """
    Generates a temporary file path by appending a suffix before the file extension.
    
    Args:
        path (str): The original target file path.
        suffix (str, optional): The suffix to append. Defaults to "tmp".
        
    Returns:
        str: The newly constructed temporary file path.
    """
    # Split the original path into base name and extension
    base, ext = os.path.splitext(path)
    # Reconstruct the path with the suffix injected before the extension
    return f"{base}_{suffix}{ext}"

class CWRUMetadataParser:
    """
    Responsible ONLY for extracting metadata from the CWRU filename.
    Separating this logic ensures the single-responsibility principle.
    """
    
    @staticmethod
    def parse(filename: str) -> Dict[str, Any]:
        """
        Parses the standardized CWRU filename to extract fault type, severity, 
        horsepower (HP), and sensor location.
        
        Args:
            filename (str): The name of the file to parse.
            
        Returns:
            Dict[str, Any]: A dictionary containing the extracted metadata.
        """
        # Remove the file extension to isolate the metadata string
        name = filename.replace(".mat", "")
        # Split the string by underscores to get individual metadata components
        parts = name.split("_")

        # Initialize a default metadata dictionary structure
        meta = {
            "fault": "Normal",
            "severity": "",
            "hp": 0,
            "loc": -1
        }

        # Check if the file represents a baseline 'Normal' condition
        if parts[0] == "Normal":
            # Extract the horsepower by stripping the "HP" string and converting to integer
            meta["hp"] = int(parts[1].replace("HP", ""))
        else:
            # Locate and extract the specific defect type (e.g., IR, OR, Ball)
            flt = next(p for p in parts if p.startswith("Defect")).replace("Defect", "")
            # Locate and extract the bearing position (e.g., DE or FE)
            bearing = next(p for p in parts if p.startswith("Bearing")).replace("Bearing", "")
            # Combine bearing and fault type if it's Fan End (FE), otherwise just use fault type
            meta["fault"] = f"{bearing}_{flt}" if bearing == 'FE' else flt

            # Locate and extract the fault diameter (severity)
            meta["severity"] = next(p for p in parts if p.startswith("Diameter")).replace("Diameter", "")

            # Locate and extract the horsepower load
            meta["hp"] = int(next(p for p in parts if "HP" in p).replace("HP", ""))

            # Look for location data in the filename parts
            loc_parts = [p for p in parts if p.startswith("Location")]
            # If a location is specified, extract and convert it to an integer
            if loc_parts:
                meta["loc"] = int(loc_parts[0].replace("Location", ""))

        # Return the fully populated metadata dictionary
        return meta


class CWRUSignalExtractor:
    """
    Responsible ONLY for loading .mat files and safely extracting DE, FE, and BA arrays.
    Handles data anomalies and corrupted files safely.
    """

    # Dictionary mapping problematic CWRU filenames to the correct signal array index
    DUPLICATE_RULES = {
        "BearingDE_DefectIR_Diameter021_3HP": 1,
        "Normal_2HP": 1,
        "BearingDE_DefectIR_Diameter014_1HP": 0
    }
    
    def __init__(self):
        """Initializes the standard key suffixes used in CWRU .mat files."""
        self.KEY_SUFFIXES = {
            "DE": "DE_time",
            "FE": "FE_time",
            "BA": "BA_time"
        }

    def extract(self, filepath: str, dtype=np.float32) -> Optional[Dict[str, np.ndarray]]:
        """
        Loads the .mat file, resolves duplicate array keys, and extracts full-length signals.
        
        Args:
            filepath (str): Full path to the .mat file.
            dtype (type, optional): Target NumPy data type for the signals. Defaults to np.float32.
            
        Returns:
            Optional[Dict[str, np.ndarray]]: A dictionary of signal arrays, or None if the file is invalid.
        """
        # Extract the base filename for logging purposes
        filename = os.path.basename(filepath)
        try:
            # Attempt to load the MATLAB dictionary file
            mat = scipy.io.loadmat(filepath)
        except Exception as e:
            # Catch file corruption or read errors and skip gracefully
            print(f"⚠️ Corrupted file skipped [{filename}]: {e}")
            return None

        # ----------------------------------------------------------------
        # Inner helper function to safely parse and resolve a single sensor
        # ----------------------------------------------------------------
        def _get_sensor_signal(sensor_name: str) -> Optional[np.ndarray]:
            """Extracts the signal array for a specific sensor type."""
            # Look up the expected suffix for the requested sensor
            suffix = self.KEY_SUFFIXES[sensor_name]
            # Find all keys in the .mat file that end with the target suffix
            matching_keys = sorted([k for k in mat.keys() if k.endswith(suffix)])
            
            # If no keys match, return None to indicate the sensor is missing
            if not matching_keys:
                return None
                
            # Extract and flatten (squeeze) the arrays for all matching keys
            raw_signals = [mat[k].squeeze() for k in matching_keys]
            # Pass the signals to the duplicate resolution handler
            return self._resolve_duplicates(raw_signals, filename, sensor_name)

        # 1. Drive End (DE) - Must exist as the reference anchor for dataset alignment
        de = _get_sensor_signal("DE")
        # If the primary DE signal is missing, reject the entire file
        if de is None:
            print(f"⚠️ Skipping {filename}: DE reference signal is missing.")
            return None
            
        # Determine the sequence length based on the DE reference signal
        signal_length = len(de)
        # Initialize the final signals dictionary with the extracted DE array
        signals = {"DE": de.astype(dtype)}

        # 2 & 3. Fan End (FE) and Base Plate (BA) - Dynamic allocation or NaN fallback
        for sensor in ["FE", "BA"]:
            # Attempt to extract the signal for the current sensor
            signal_data = _get_sensor_signal(sensor)
            if signal_data is not None:
                # If found, cast to the target dtype and store it
                signals[sensor] = signal_data.astype(dtype)
            else:
                # If missing, pad with NaN values matching the DE signal length
                signals[sensor] = np.full(signal_length, np.nan, dtype=dtype)

        # Return the complete dictionary of extracted signals
        return signals

    @classmethod
    def _resolve_duplicates(cls, signals: List[np.ndarray], filename: str, sensor_type: str) -> np.ndarray:
        """
        Handles edge cases where CWRU .mat files contain multiple array keys.
        
        Args:
            signals (List[np.ndarray]): List of arrays matching the sensor type.
            filename (str): Name of the file being processed.
            sensor_type (str): The type of sensor (e.g., 'DE', 'FE').
            
        Returns:
            np.ndarray: The correct signal array based on predefined rules.
        """
        # If there is only one signal, no resolution is needed
        if len(signals) == 1:
            return signals[0]

        # Log a warning that duplicates were found
        print(f"🟡 Duplicate {sensor_type} signals in {filename} -> Applying resolution rules.")
        
        # Strip .mat extension for clean dictionary lookup in DUPLICATE_RULES
        clean_name = filename.replace(".mat", "")
        
        # Check if we have a predefined manual rule for this exact file
        if clean_name in cls.DUPLICATE_RULES:
            # Apply the manual rule to select the correct array index
            target_index = cls.DUPLICATE_RULES[clean_name]
            return signals[target_index]
        
        # Fallback warning if an unknown anomaly is encountered in the dataset
        print(f"🔍 Unknown duplicate pattern in {filename}. Defaulting to index 0.")
        return signals[0]
    

class CWRUIngestor:
    """
    Orchestrates parallel file processing and aligned saving to a single .npz file.
    Acts as the main pipeline manager for the ingestion phase.
    """
    
    def __init__(self, dtype = np.float32):
        """Initializes the ingestor with expected sensors and data precision."""
        # Standard sensor list used across the CWRU dataset
        self.sensors = ["DE", "FE", "BA"]
        # Set the target data type for numerical stability and memory efficiency
        self.dtype = dtype
        # Instantiate the specialized signal extractor
        self.signal_extractor = CWRUSignalExtractor()

    def _process_single_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Worker function for the thread pool to process an individual file.
        
        Args:
            file_path (str): Path to the .mat file.
            
        Returns:
            Optional[Dict[str, Any]]: A combined dictionary of metadata and signals, or None on failure.
        """
        # Extract the base filename from the path
        filename = os.path.basename(file_path)
        
        # 1. Parse Metadata using the dedicated parser
        meta = CWRUMetadataParser.parse(filename)
        
        # 2. Extract Signals using the dedicated extractor
        signals = self.signal_extractor.extract(file_path, dtype=self.dtype)
        
        # If signal extraction failed (e.g., missing DE), return None
        if signals is None:
            return None
            
        # Combine the metadata and signal dictionaries into a single flat record
        return {**meta, **signals, "filename": filename}

    def ingest_directory(
        self, 
        base_path: str = 'data/CWRU12/RawFiles', 
        output_file_path: str = 'data/CWRU12/CWRU12Ingested', 
        max_workers: int = 8,
        replace_file: bool = False
    ) -> bool:
        """
        Reads all .mat files in parallel and saves a unified, aligned .npz file.
        
        Args:
            base_path (str): Directory containing raw .mat files.
            output_file_path (str): Target path for the output .npz file.
            max_workers (int): Number of parallel threads to spawn.
            replace_file (bool): Whether to overwrite an existing database.
            
        Returns:
            bool: True if ingestion finishes successfully, False otherwise.
        """
        # Ensure the output path has the correct .npz extension
        if not output_file_path.endswith('.npz'):
            output_file_path += ".npz"
            
        # If the file exists and replace_file is False, skip ingestion
        if os.path.exists(output_file_path) and not replace_file:
            print(f"⏭️ Output file {output_file_path} already exists. Skipping ingestion.")
            return True

        # Gather all .mat files from the specified base directory
        mat_files = [
            os.path.join(base_path, f) 
            for f in os.listdir(base_path) 
            if f.endswith(".mat")
        ]
        # Sort the files alphabetically for consistent processing order
        mat_files.sort()

        # Abort if no files are found in the directory
        if not mat_files:
            print(f"⚠️ No .mat files found in {base_path}.")
            return False

        # Initialize the main storage dictionary with empty lists for column-oriented storage
        dataset = {
            "filename": [], "fault": [], "severity": [], "hp": [], "loc": []
        }
        # Add dynamic storage lists for each sensor type
        for s in self.sensors:
            dataset[s] = []

        print(f"🚀 Ingesting {len(mat_files)} files in parallel using {max_workers} workers...")
        
        # Execute the processing function concurrently using a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files to the executor and map futures to their file paths
            future_to_path = {executor.submit(self._process_single_file, path): path for path in mat_files}
            
            # Iterate through futures as they complete, wrapping with a tqdm progress bar
            for future in tqdm(as_completed(future_to_path), total=len(mat_files), desc="Processing .mat files"):
                # Retrieve the processing result
                record = future.result()
                # If the record is valid, append its contents to the dataset columns
                if record is not None:
                    # Append metadata strictly maintaining row-level alignment
                    dataset["filename"].append(record["filename"])
                    dataset["fault"].append(record["fault"])
                    dataset["severity"].append(record["severity"])
                    dataset["hp"].append(record["hp"])
                    dataset["loc"].append(record["loc"])
                    # Append signal data dynamically for all tracked sensors
                    for s in self.sensors:
                        dataset[s].append(record[s])

        # Calculate the final number of successfully processed files
        valid_records = len(dataset["filename"])
        print(f"✅ Extracted {valid_records} valid records out of {len(mat_files)} files.")

        # Save the populated dataset dictionary to the target .npz archive
        self._save_unified_npz(dataset, output_file_path)
        return True

    def _save_unified_npz(self, dataset: Dict[str, List[Any]], npz_path: str):
        """
        Converts Python lists to appropriate NumPy arrays and saves safely to disk.
        
        Args:
            dataset (Dict[str, List[Any]]): The fully populated dataset dictionary.
            npz_path (str): The final destination path for the .npz archive.
        """
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        # Create a temporary path to avoid corrupting data if the save is interrupted
        temp_path = _get_temp_path(npz_path)

        # Convert metadata lists to typed NumPy arrays for efficient storage and querying
        save_dict = {
            "Filename": np.array(dataset["filename"], dtype=str),
            "Fault": np.array(dataset["fault"], dtype=str),
            "Severity": np.array(dataset["severity"], dtype=str),
            "HorsePower": np.array(dataset["hp"], dtype=np.int32),
            "Location": np.array(dataset["loc"], dtype=np.int32),
        }

        # Convert signal lists to NumPy object arrays (since signal lengths vary per file)
        for s in self.sensors:
            save_dict[s] = np.array(dataset[s], dtype=object)

        # Execute the write operation to the temporary file path
        print(f"💾 saving data to {npz_path}...")
        np.savez(temp_path, **save_dict)
        
        # Atomically replace the temporary file with the final target file path
        os.replace(temp_path, npz_path)
        print(f"🎉 Ingestion complete! Unified dataset saved at: {npz_path}")