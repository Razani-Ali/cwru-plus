# cwru/__init__.py

"""
CWRU Plus: An enterprise-grade, ultra-fast pipeline for downloading,
ingesting, filtering, and preparing the Case Western Reserve University (CWRU)
bearing dataset for Machine Learning and Deep Learning workloads.
"""

import sys
import numpy as np
from typing import Callable, Dict, Tuple, Union, List, Optional

from .downloader import download_CWRU_files
from .ingestion import CWRUIngestor
from .filtering import DatasetFilter
from .dataset import CWRUDatasetBuilder

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
                - (X, Y): 
                    X: Extracted windows with shape (Samples, Channels, Window_Size) if num_parts is None,
                       or (Num_Parts, Samples_per_Part, Channels, Window_Size) if num_parts is active.
                    Y: Target string labels matching the configuration strategy.
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

# Bind the pipeline object directly to the module namespace for a seamless user experience
sys.modules[__name__] = CWRUPipeline()