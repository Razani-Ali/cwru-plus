import os
import numpy as np
from typing import Callable, Dict, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def _get_temp_path(path: str, suffix: str = "tmp") -> str:
    """
    Generates a temporary file path to ensure safe, atomic file saving.
    
    Args:
        path (str): The target destination file path.
        suffix (str, optional): The suffix to append before the extension. Defaults to "tmp".
        
    Returns:
        str: The constructed temporary file path.
    """
    # Ensure the path explicitly has an .npz extension before splitting
    if not path.endswith('.npz'):
        path += '.npz'
    # Split the path into the base directory/filename and the extension
    base, ext = os.path.splitext(path)
    # Reconstruct the path with the temporary suffix injected
    return f"{base}_{suffix}{ext}"


class DatasetFilter:
    """
    Orchestrates parallel application of user-defined filter functions to the dataset.
    Supports channel-specific filtering (e.g., different filters for DE and FE sensors).
    """
    
    def __init__(self, filters: Union[Callable, Dict[str, Callable]],
                 dtype = np.float32):
        """
        Initializes the DatasetFilter with the provided signal processing functions.
        
        Args:
            filters: A single callable (applied to all sensors) OR a dictionary mapping 
                     sensor names to specific callables (e.g., {"DE": func1, "FE": func2}).
            dtype: The target NumPy data type for the filtered arrays. Defaults to np.float32.
        """
        # Define the standard list of sensors expected in the CWRU dataset
        self.sensors = ["DE", "FE", "BA"]
        # Set the target data type for memory efficiency
        self.dtype = dtype
        
        # If the user provides a single function, duplicate it for all defined sensors
        if callable(filters):
            self.filters = {s: filters for s in self.sensors}
        # If the user provides a dictionary, use it directly for channel-specific logic
        elif isinstance(filters, dict):
            self.filters = filters
        # Reject invalid filter inputs
        else:
            raise TypeError("Filters must be a single callable or a dictionary of callables.")

    def _process_single_record(self, record_idx: int, data_dict: Dict[str, Any]) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Worker function: Applies the specific filter to each sensor for a single dataset record.
        
        Args:
            record_idx (int): The index of the row being processed.
            data_dict (Dict[str, Any]): The full materialized dataset dictionary.
            
        Returns:
            Tuple[int, Dict[str, np.ndarray]]: The index of the record and its filtered sensor arrays.
        """
        # Dictionary to hold the newly processed signals for this specific record
        filtered_signals = {}
        
        # Iterate over all defined standard sensors
        for sensor in self.sensors:
            # Skip if the target sensor does not exist in the dataset
            if sensor not in data_dict:
                continue
                
            # Extract the raw signal array for this specific sensor and record index
            original_signal = data_dict[sensor][record_idx]
            
            # 1. If the sensor is not in the filters dictionary, or its assigned filter is None,
            # bypass processing and return the raw original signal.
            if sensor not in self.filters or self.filters[sensor] is None:
                filtered_signals[sensor] = original_signal
                continue
                
            # 1.1 Check if the signal is completely empty or consists entirely of NaN values.
            # If so, return it as-is without attempting to apply mathematical filters.
            if original_signal.size == 0 or np.isnan(original_signal).all():
                filtered_signals[sensor] = original_signal
                continue

            # 2. Retrieve the specific user-defined filter function for this channel
            filter_func = self.filters[sensor]
            try:
                # Execute the filter function on the original 1D signal
                processed = filter_func(original_signal)
                # Cast the output back to the target NumPy dtype and store it
                filtered_signals[sensor] = np.asarray(processed, dtype=self.dtype)
            except Exception as e:
                # Catch and raise any errors caused by the user's custom filter logic
                raise RuntimeError(f"Error applying filter on {sensor} for record {record_idx}: {e}")
                
        # Return the row index alongside the newly computed dictionary to maintain alignment
        return record_idx, filtered_signals

    def process_file(self, input_npz: str = 'data/CWRU12/CWRU12Ingested.npz',
                     output_npz: str = 'data/CWRU12/CWRU12IngestedFilter',
                     max_workers: int = 8) -> bool:
        """
        Loads the ingested data, applies channel-specific filters in parallel, and saves the output.
        
        Args:
            input_npz (str): Path to the raw/ingested .npz file.
            output_npz (str): Target destination path for the filtered .npz file.
            max_workers (int): Number of parallel CPU threads to utilize.
            
        Returns:
            bool: True if the file was successfully processed and saved.
        """
        # Validate that the source file actually exists before proceeding
        if not os.path.exists(input_npz):
            raise FileNotFoundError(f"❌ Input file not found: {input_npz}")
        
        # Ensure the output path ends with the correct .npz extension
        if not output_npz.endswith(".npz"):
            output_npz += ".npz"
            
        print(f"📦 Loading dataset from {input_npz}...")
        # Load the raw dataset into memory
        data = np.load(input_npz, allow_pickle=True)
        # Determine the total number of records (rows) by checking the Filename array length
        num_records = len(data["Filename"])

        # Copy all structural metadata (e.g., Fault, Severity, Location) into the output dictionary
        # by excluding keys that match sensor names.
        out_data = {key: data[key] for key in data.files if key not in self.sensors}
        
        # Prepare empty lists in the output dictionary for the filtered sensor data
        # Pre-allocating lists to the exact size of num_records ensures strict row alignment
        for s in self.sensors:
            if s in data:
                out_data[s] = [None] * num_records

        print(f"🚀 Filtering {num_records} records in parallel using {max_workers} workers...")
        
        # Instantiate a parallel thread pool executor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all row indices to the executor, mapping each Future to its index
            future_to_idx = {
                executor.submit(self._process_single_record, i, data): i 
                for i in range(num_records)
            }
            
            # Wrap the parallel completion loop in a tqdm progress bar for visual tracking
            for future in tqdm(as_completed(future_to_idx), total=num_records, desc="Applying Filters"):
                # Retrieve the specific index and its newly processed dictionary
                idx, filtered_sigs = future.result()
                # Slot the filtered arrays back into their exact pre-allocated positions
                for s, filtered_arr in filtered_sigs.items():
                    out_data[s][idx] = filtered_arr

        # Convert the populated Python lists back into NumPy object arrays 
        # (Object arrays are required because signal lengths can vary dynamically)
        for s in self.sensors:
            if s in data:
                out_data[s] = np.array(out_data[s], dtype=object)

        # Re-verify the output path extension (failsafe)
        if not output_npz.endswith('.npz'):
            output_npz += '.npz'
            
        # Generate the temporary file path
        temp_path = _get_temp_path(output_npz)
        # Ensure the parent directory structure exists for the target destination
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)
        
        print(f"💾 Saving filtered dataset to {output_npz}...")
        # Save the finalized dictionary to the temporary path using Zlib compression
        np.savez_compressed(temp_path, **out_data)
        # Atomically replace the target file with the temporary file
        os.replace(temp_path, output_npz)
        
        print("🎉 Filtering complete!")
        return True
    
# ----------------------------------------------------------------
# Example usage demonstrating dependency injection for filters:
# ----------------------------------------------------------------
# import scipy.signal as sig
# # Design a 4th-order low-pass Butterworth filter
# b, a = sig.butter(4, 0.1, btype='low')
# # Define a zero-phase filter function (avoids phase shift)
# zero_phase_func = lambda x: sig.filtfilt(b, a, x)
# # Define a causal forward filter function
# causal_filter_func = lambda x: sig.lfilter(b, a, x)