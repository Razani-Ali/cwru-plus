# cwru/__init__.py

"""
CWRU Plus: An enterprise-grade, ultra-fast pipeline for downloading,
ingesting, filtering, and preparing the Case Western Reserve University (CWRU)
bearing dataset for Machine Learning and Deep Learning workloads.
"""

import sys
import numpy as np
from typing import Callable, Dict, Tuple, Union, List, Optional

from .downloader import download_CWRU_files, local_download
from .ingestion import CWRUIngestor
from .filtering import DatasetFilter
from .dataset import CWRUDatasetBuilder, generate_stratified_file_split
from .fewshot_sampler import FewShotSampler

__version__ = "0.1.0"

# --------------------------------------------------------------------
# Define a clean high-level interface (Facade Pattern)
# --------------------------------------------------------------------

class CWRUPipeline:
    """High-level API wrapping dataset operations for end-users.
    
    Acts as a unified facade for concurrently downloading raw data, parsing MAT files,
    parallel filtering, and lightning-fast temporal/sliding-window data construction.
    """
    
    @staticmethod
    def download(
        CWRUfs: int = 48,
        download_path: str = 'data/CWRU12/RawFiles',
        max_retries: int = 10,
        retry_delay: int = 3,
        replace_files: bool = False,
        max_workers: int = 8
    ) -> bool:
        """Downloads the raw CWRU dataset files concurrently from the Case Western server.

        Args:
            CWRUfs (int, optional): Target sampling frequency in kHz. Options are 12 or 48. Defaults to 48.
            download_path (str, optional): Local directory where raw .mat files will be saved. Defaults to 'data/CWRU12/RawFiles'.
            max_retries (int, optional): Number of times to retry a failed download block before giving up. Defaults to 10.
            retry_delay (int, optional): Wait time (seconds) between successive download retries. Defaults to 3.
            replace_files (bool, optional): If True, re-downloads and overwrites existing files. Defaults to False.
            max_workers (int, optional): Maximum number of concurrent threads used for downloading. Defaults to 8.

        Returns:
            bool: True if all target files are successfully downloaded and verified, False otherwise.
        """
        return download_CWRU_files(
            download_path=download_path,
            max_retries=max_retries,
            retry_delay=retry_delay,
            CWRUfs=CWRUfs,
            replace_files=replace_files,
            max_workers=max_workers
        )
    
    @staticmethod
    def offline_download(
        source_path: str,
        target_path: str = 'data/CWRU/RawFiles',
        CWRUfs: int = 12,
        replace_files: bool = False
    ) -> bool:
        """
        Exposes a clean, high-level API to import, parse, and standardize manually 
        downloaded CWRU MATLAB (.mat) files from a local directory.

        This function cross-references the files inside the `source_path` against 
        the official structural grid URLs of the requested sampling frequency. It selectively 
        isolates the correct files, maps them to a highly consistent, machine-learning-ready 
        naming convention, and duplicates them into the `target_path`. 

        To prevent data corruption or conflicts, your original source files are left 
        completely unmodified. Files belonging to other sampling rates in the same directory 
        are automatically ignored unless requested.

        Args:
            source_path (str): Absolute or relative path to the local directory where 
                            the raw, manually downloaded `.mat` files (e.g., '105.mat') reside.
            target_path (str): The destination directory where the newly renamed and 
                            standardized files will be stored. Defaults to 'data/CWRU/RawFiles'.
            CWRUfs (int): The target sampling frequency to filter and build (Must be either 12 or 48). 
                        Defaults to 12.
            replace_files (bool): If True, overwrites existing files in the destination directory. 
                                If False, skips processing for already existing standardized files. 
                                Defaults to False.

        Returns:
            bool: True if matching files were successfully identified, mapped, and copied to 
                the target directory; False if no compatible source files were found.

        Raises:
            ValueError: If `CWRUfs` is passed an invalid integer other than 12 or 48.
        """

        # Simply delegate the args to the underlying local implementation
        return local_download(
            source_path=source_path,
            target_path=target_path,
            CWRUfs=CWRUfs,
            replace_files=replace_files
        )

    @staticmethod
    def ingest(
        base_path: str = 'data/CWRU12/RawFiles',
        output_file_path: str = 'data/CWRU12/CWRU12Ingested',
        max_workers: int = 8,
        replace_file: bool = False,
        dtype = np.float32,
    ) -> bool:
        """Ingests raw .mat files in parallel and packs them into a single unified .npz database.

        Extracts structured metadata (Fault, Severity, Location, HorsePower) from filenames 
        and structural paths, normalizing signal arrays across channels.

        Args:
            base_path (str, optional): Directory containing the raw downloaded .mat files. Defaults to 'data/CWRU12/RawFiles'.
            output_file_path (str, optional): Base output path for the processed database file (without .npz extension). Defaults to 'data/CWRU12/CWRU12Ingested'.
            max_workers (int, optional): Number of parallel CPU workers used to parse files. Defaults to 8.
            replace_file (bool, optional): If True, overwrites an existing ingested database. Defaults to False.
            dtype (numpy.dtype, optional): Precision data-type used to save continuous vibration signals. Defaults to np.float32.

        Returns:
            bool: True upon successful validation and safe atomic replacement of the .npz archive.
        """
        ingestor = CWRUIngestor(dtype=dtype)
        return ingestor.ingest_directory(
            base_path=base_path,
            output_file_path=output_file_path,
            max_workers=max_workers,
            replace_file=replace_file
        )
    
    @staticmethod
    def filter(
        filters: Union[Callable[[np.ndarray], np.ndarray], Dict[str, Callable[[np.ndarray], np.ndarray]]],
        input_npz: str = 'data/CWRU12Ingested.npz',
        output_npz: str = 'data/ingested12filter',
        max_workers: int = 8
    ) -> bool:
        """Applies custom linear or non-linear signal processing filters to the dataset in parallel.

        Supports dynamic channel-specific inversion of control (IoC). You can pass a single callable
        or a dictionary mapping specific channels to unique functions.

        Args:
            filters (callable or dict): A single function applied to all sensors, OR a dictionary 
                mapping sensor names to functions (e.g., {"DE": lambda x: scipy.signal.medfilt(x), "FE": None}).
            input_npz (str, optional): Path to the source ingested .npz archive. Defaults to 'data/ingested12.npz'.
            output_npz (str, optional): Path where the new filtered database will be written. Defaults to 'data/ingested12filter'.
            max_workers (int, optional): Number of concurrent execution threads. Defaults to 8.

        Returns:
            bool: True if all records are successfully processed and written to disk.
        """
        filt = DatasetFilter(filters=filters)
        return filt.process_file(
            input_npz=input_npz,
            output_npz=output_npz,
            max_workers=max_workers
        )
    
    @staticmethod
    def load(
        npz_path: str = 'data/CWRU12/CWRU12Ingested.npz',
        window_size: int = 2048, 
        step_size: int = 512, 
        sensors: List[str] = ["DE", "FE"], 
        strategy: str = "standard", 
        fault_category: str = "types & severity",
        num_parts: Optional[int] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
        """Loads data into RAM and builds ultra-fast training tensors with optional temporal partitioning.

        Leverages zero-copy memory views via NumPy stride tricks to extract overlapping sliding windows
        in milliseconds. Eliminates Python list overhead via full vectorized data replication.

        Args:
            npz_path (str, optional): Path to the processed .npz database file. Defaults to 'data/CWRU12/CWRU12Ingested.npz'.
            window_size (int, optional): Number of vibration datapoints inside each sample window. Defaults to 2048.
            step_size (int, optional): The stride size for sliding the extraction window. Defaults to 512.
            sensors (List[str], optional): Selected continuous channels to include in the output channel dimension. Defaults to ["DE", "FE"].
            strategy (str, optional): Filtering domain constraints. Options are:
                - 'standard': Only normal and Drive End (DE) faults at standard loads/severities.
                - 'extended': Adds Fan End (FE) bearing faults.
                - 'full': Loads all rows unconditionally.
                Defaults to 'standard'.
            fault_category (str, optional): Strategy for target label creation. Options are:
                - 'only types': Strings containing core category (e.g., 'IR', 'Ball', 'Normal').
                - 'types & severity': Combined categories (e.g., 'IR007', 'OR021').
                - 'types, severity & locations': Full detailed structural metadata (e.g., 'OR014@3').
                Defaults to 'types & severity'.
            num_parts (int, optional): Number of equal temporal chunks to divide each original signal into
                before windowing. If set, adds a new 0-th fold dimension to arrays (useful for time-series validation). Defaults to None.

        Returns:
            Tuple: A pair of structured tuples:
                - (X, Y, ID): 
                    X: Extracted windows with shape (Samples, Channels, Window_Size) if num_parts is None,
                       or (Num_Parts, Samples_per_Part, Channels, Window_Size) if num_parts is active.
                    Y: Target string labels matching the configuration strategy.
                    ID: File Index Related to Extracted Windows, Useful for File Based Data Separation
                - (Severities, HorsePowers, Locations):
                    Vectorized metadata tracking original structural records for each generated window.
        """
        builder = CWRUDatasetBuilder(npz_path)
        return builder.build(
            window_size=window_size,
            step_size=step_size,
            sensors=sensors,
            strategy=strategy,
            fault_category=fault_category,
            num_parts=num_parts
        )

    @staticmethod
    def stratified_file_split(
        X: np.ndarray, 
        y: np.ndarray, 
        file_ids: np.ndarray, 
        train_ratio: float = 0.75, 
        val_ratio: float = 0.25, 
        random_seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        A high-level wrapper that automates leakage-free, stratified data splitting 
        and returns ready-to-use NumPy arrays via Fancy Indexing.

        This function wraps `generate_stratified_file_split` to simplify the user pipeline.
        It splits the dataset in a single call while ensuring that:
        1. No physical file spans across different splits (Zero Data Leakage).
        2. The target class distribution remains identical in Train, Val, and Test sets.

        Args:
            X (np.ndarray): The feature matrix/tensor containing signal windows, shape [N, window_size].
            y (np.ndarray): The 1D target array containing class labels, shape [N].
            file_ids (np.ndarray): The 1D array identifying the source file of each window, shape [N].
            train_ratio (float): Percentage of physical files allocated to Training (default: 0.8).
            val_ratio (float): Percentage of physical files allocated to Validation (default: 0.1).
            random_seed (int): Control seed for shuffling reproducibility (default: 42).

        Returns:
            Tuple[Tuple, Tuple]: A nested tuple containing two main components:
                1. Data Splits: (train_data, val_data, test_data) where each is a tuple of (X_split, y_split).
                2. Indices Splits: (train_idx, val_idx, test_idx) containing the raw NumPy index arrays.
        """
        # 1. Calling the core stratified file-based index generator
        train_idx, val_idx, test_idx = generate_stratified_file_split(
            y=y,
            file_ids=file_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=random_seed
        )

        train_d = (X[train_idx], y[train_idx])
        val_d = (X[val_idx], y[val_idx])
        test_d = (X[test_idx], y[test_idx])
        
        return (train_d, val_d, test_d), (train_idx, val_idx, test_idx)

    @staticmethod
    def build_few_shot_sampler(
        X_base: np.ndarray,
        Y_base: np.ndarray,
        numeric_to_string: Dict[int, str],
        meta_base: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (None, None, None),
        seed: Optional[int] = None
    ): # -> FewShotSampler
        """
        Initializes and returns a high-performance FewShotSampler instance for pre-windowed datasets.
        
        This wrapper abstracts the instantiation process, allowing seamless integration into 
        data loading pipelines. The returned sampler object can be queried repeatedly via 
        its `.sample()` method to extract isolated episodic tasks for meta-learning.

        Args:
            X_base (np.ndarray): The pre-windowed 3D input tensor of shape [Total_Windows, Channels, Length].
            Y_base (np.ndarray): Master 1D array of original string labels corresponding to each window.
            numeric_to_string (Dict[int, str]): Dictionary mapping PyTorch-compatible integer classes 
                                                to database string names (e.g., {0: 'Normal', 1: 'Inner_Fault'}).
            meta_base (Tuple[np.ndarray, np.ndarray, np.ndarray], optional): 
                Bound operational metadata arrays corresponding to each window instance.
                Expected format: (Severity_Array, HP_Array, Location_Array). 
                Defaults to (None, None, None).
            seed (int, optional): Pseudo-random generator state for exact task reproducibility.

        Returns:
            FewShotSampler: An initialized sampler object ready for `.sample()` queries.
        """
        # Initialize the core sampler using the pre-processed 3D tensors
        sampler = FewShotSampler(
            X_base=X_base, 
            Y_base=Y_base, 
            numeric_to_string=numeric_to_string,
            meta_base=meta_base
        )
        
        # Lock the seed if provided for reproducible episodic generation
        if seed is not None:
            sampler.reset_seed(seed)
            
        return sampler

# Bind the pipeline object directly to the module namespace for a seamless user experience
sys.modules[__name__] = CWRUPipeline()