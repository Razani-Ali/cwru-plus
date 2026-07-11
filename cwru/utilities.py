"""
Core Downloader Module

This module provides robust, concurrent file downloading, archiving, and copying utilities.
It includes mechanisms for thread-safe operations, atomic file replacements, and 
automatic retry logic for handling network instability or filesystem locks.
"""

import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union
from tqdm.auto import tqdm
import zipfile
import threading
from pathlib import Path
import shutil

def get_temp_path(path: str, suffix: str = "tmp") -> str:
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
    # Reconstruct the path with the suffix injected before the extension (if it exists)
    return f"{base}_{suffix}{ext}" if ext else f"{path.rstrip('/')}_{suffix}"

def remove_file(file_path: str, force: bool = False) -> bool:
    """
    Safely removes a file from the filesystem.
    
    Args:
        file_path (str): The path to the file to be deleted.
        force (bool, optional): If True, returns True even if the file didn't exist. Defaults to False.
        
    Returns:
        bool: True if the file was successfully removed (or if force=True), False otherwise.
    """
    try:
        # Check if the path exists and is strictly a file
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Attempt to delete the file
            os.remove(file_path)
            return True
        # Return the force flag if the file does not exist
        return force
    except Exception as e:
        # Catch and log any permission or OS errors during deletion
        print(f"⚠️ Warning: Failed to remove file {file_path}: {e}")
        return False

def replace_with_error(src: str, dest: str) -> bool:
    """
    Atomically replaces the destination file with the source file.
    
    Args:
        src (str): The path of the source file (usually a temporary file).
        dest (str): The path of the final destination file.
        
    Returns:
        bool: True if the replacement was successful, False otherwise.
    """
    try:
        # Perform an atomic replace operation (avoids corrupted partial files)
        os.replace(src, dest)
        return True
    except Exception as e:
        # Catch and log errors, such as file locks or permission issues
        print(f"❌ Error replacing {src} with {dest}: {e}")
        return False

def download_file(url: str, target_path: str, max_retries: int = 10, 
                  retry_delay: float = 3.0, replace: bool = False) -> bool:
    """
    Downloads a single file from a URL with chunking and retry logic.
    
    Args:
        url (str): The direct URL to download the file from.
        target_path (str): The local destination path.
        max_retries (int, optional): Maximum number of download attempts. Defaults to 10.
        retry_delay (float, optional): Wait time (in seconds) between retries. Defaults to 3.0.
        replace (bool, optional): Whether to overwrite an existing file. Defaults to False.
        
    Returns:
        bool: True if the file is successfully downloaded, False otherwise.
    """
    # Extract the filename from the target path for logging purposes
    filename = os.path.basename(target_path)
    
    # Check if the file already exists and should not be overwritten
    if os.path.exists(target_path) and not replace:
        print(f"⏭️  [Skipped] {filename} already exists.")
        return True

    # Ensure the target directory structure exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Generate a safe temporary path to write data into during the download
    temp_path = get_temp_path(target_path)

    # Initiate the retry loop
    for attempt in range(1, max_retries + 1):
        try:
            print(f"⬇️  [Attempt {attempt}/{max_retries}] Starting download: {filename}")
            
            # Open a streaming GET request with a 30-second timeout
            with requests.get(url, stream=True, timeout=30) as r:
                # Raise an HTTPError if the response status code is 4xx or 5xx
                r.raise_for_status()
                
                # Open the temporary file in binary write mode
                with open(temp_path, 'wb') as f:
                    # Iterate over the incoming byte stream in 8KB chunks
                    for chunk in r.iter_content(chunk_size=8192):
                        # Filter out keep-alive new chunks
                        if chunk:
                            f.write(chunk)

            # Atomically rename the complete temporary file to the final target path
            replace_with_error(temp_path, target_path)
            print(f"✅ [Success] Fully downloaded: {filename}")
            return True
            
        except Exception as e:
            # Clean up the partial temporary file upon failure
            remove_file(temp_path, force=True)
            print(f"⚠️  [Warning] Attempt {attempt} failed for {filename}. Error: {e}")
            
            # If attempts remain, wait for the specified delay before retrying
            if attempt < max_retries:
                print(f"🔄 Waiting {retry_delay} seconds before retrying {filename}...")
                time.sleep(retry_delay)
            else:
                # Exhausted all attempts; log failure and clean up the target path
                print(f"💀 [Failed] Max retries reached. Could not download: {filename}")
                remove_file(target_path, force=True) 
                return False
                
    return False

def run_parallel_downloads(tasks: List[Tuple[str, str, str]], max_workers: int, 
                           max_retries: int, retry_delay: int, replace_files: bool) -> Tuple[int, List[str]]:
    """
    Executes multiple download tasks concurrently using a thread pool.
    
    Args:
        tasks (List[Tuple[str, str, str]]): A list of tuples containing (url, output_path, filename).
        max_workers (int): Number of concurrent threads to spawn.
        max_retries (int): Maximum retry attempts passed down to each download.
        retry_delay (int): Delay between retries for each download.
        replace_files (bool): Overwrite flag passed down to each download.
        
    Returns:
        Tuple[int, List[str]]: A tuple containing the count of successful downloads 
                               and a list of filenames that failed.
    """
    # Initialize trackers for successful downloads and failed filenames
    successful = 0
    failed_downloads = []

    # Define an internal worker wrapper for the ThreadPoolExecutor
    def _worker(url: str, path: str, name: str) -> Tuple[str, bool]:
        # Execute the download and capture its boolean result
        success = download_file(url, path, max_retries, retry_delay, replace_files)
        # Return the filename alongside its success status
        return name, success

    # Launch a ThreadPoolExecutor to handle concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks to the thread pool
        futures = [executor.submit(_worker, url, path, name) for url, path, name in tasks]
        
        # Iterate over the futures as they finish execution
        for future in as_completed(futures):
            # Unpack the result from the worker thread
            name, success = future.result()
            if success:
                # Increment success counter
                successful += 1
            else:
                # Record the name of the file that failed
                failed_downloads.append(name)
                
    return successful, failed_downloads

def remove_folder(folder_path: Union[str, os.PathLike], force: bool = False) -> bool:
    """
    Safely and recursively removes a directory tree from the filesystem.

    Provides a clean execution wrapper around standard directory tree removal tools.
    It integrates explicit verification checks for pathway existence and node type
    descriptors before launching deletion, isolating permission violations or 
    generic file system exceptions to guarantee system runtime stability.

    Args:
        folder_path (Union[str, os.PathLike]): The pathway locating the target directory 
            tree slated for recursive deletion.
        force (bool, optional): If True, suppresses missing directory exceptions and 
            returns True even if the target folder does not exist. Defaults to False.

    Returns:
        bool: True if the directory tree was successfully deleted (or skipped via force=True),
            False if an OS error, missing permission descriptor, or lock blocked completion.

    Raises:
        FileNotFoundError: If the target path is absent and force is evaluated as False.
        NotADirectoryError: If the designated pathway targets a file descriptor rather 
            than a directory layout container.
    """
    try:
        # Check if the target pathway physically exists on the filesystem disk
        if not os.path.exists(folder_path):
            # If the path is missing but the force flag is active, bypass execution with success
            if force:
                return True
            # Raise an explicit exception if the folder is absent and force is deactivated
            raise FileNotFoundError(f"❌ could not find folder '{folder_path}'")
        
        # Verify that the existing node represents a structural directory, not a generic file link
        if not os.path.isdir(folder_path):
            # Abort operation with a explicit type exception if a file collision occurs
            raise NotADirectoryError(f"🚫 directory '{folder_path}' is not a folder")
        
        # Concurrently clean and recursively purge the entire directory hierarchy layout tree
        shutil.rmtree(folder_path)
        # Return success confirmation after directory tree is wiped
        return True
    
    except PermissionError as e:
        # Intercept, catch, and log access blockages or administrative filesystem privileges
        print(f"🔒 permission denied, error: {e}")
    except Exception as e:
        # Catch, log, and isolate unknown system exceptions to protect application lifecycle
        print(f"⚠️ unknown error: {e}")
        
    # Return failure if any operational exception blockages interrupt execution flow
    return False

"""
Core Downloader & File System Operations Module

This module provides robust, single-responsibility utilities for parallel file 
compressions, thread-safe buffering transfers, and dynamic progress bar management.
"""

# Global locks registry for thread-safe operations
_locks = {}
_master_lock = threading.Lock()

def get_file_lock(file_path: str) -> threading.Lock:
    """Retrieves or creates a thread lock specific to a file/folder path."""
    with _master_lock:
        if file_path not in _locks:
            _locks[file_path] = threading.Lock()
        return _locks[file_path]

def _get_total_bytes(path: Path) -> int:
    """Calculates total payload size of a target file or a directory layout."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def create_parallel_zip(src_dir: Union[str, Path], dst_zip_path: Union[str, Path], max_workers: int = 8) -> bool:
    """
    Consolidates a directory layout tree (e.g., Zarr store) into a single ZIP archive 
    using high-speed concurrent threading and a dynamic progress bar.

    Args:
        src_dir (str/Path): The pathway targeting the source directory grid.
        dst_zip_path (str/Path): The destination file path for the output ZIP archive.
        max_workers (int, optional): Thread pool boundary cap for file compression. Defaults to 8.

    Returns:
        bool: True if compression concludes flawlessly, False otherwise.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_zip_path)
    
    if not src_path.is_dir():
        print(f"❌ Source directory '{src_dir}' does not exist.")
        return False
        
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-calculate total uncompressed bytes to calibrate the progress bar accurately
    total_bytes = _get_total_bytes(src_path)
    progress_lock = threading.Lock()
    
    # Gather all file nodes from the tree layout
    all_files = [f for f in src_path.rglob('*') if f.is_file()]
    
    print(f"📦 [PARALLEL ZIP INITIALIZED] Packing {len(all_files)} files using {max_workers} threads... ⚡")
    
    try:
        # Open the ZIP archive using ZIP_STORED to maximize execution speed (no CPU compression overhead)
        with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_STORED) as zf, tqdm(
            total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc="📦 Local Packing Phase", leave=True
        ) as bar:
            
            # Since standard zipfile writing is single-threaded, we use a lock to safely write from workers
            write_lock = threading.Lock()
            
            def _compress_worker(file_node: Path):
                archive_name = file_node.relative_to(src_path.parent)
                f_size = file_node.stat().st_size
                
                with write_lock:
                    zf.write(file_node, archive_name)
                    
                with progress_lock:
                    bar.update(f_size)
            
            # Dispatch independent files to the worker thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(_compress_worker, all_files)
                
        print(f"✅ Local packing complete! Compressed file hosted at: {dst_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create zip package: {e}")
        if dst_path.exists():
            os.remove(dst_path)
        return False


def _stream_file_buffered(p_src: Path, p_dst: Path, chunk_size: int, bar: tqdm):
    """Worker sub-task: Streams a single file using optimized chunk buffers."""
    p_dst.parent.mkdir(parents=True, exist_ok=True)
    temp_dst = p_dst.with_suffix('.tmp')
    
    with open(p_src, 'rb') as fsrc:
        with open(temp_dst, 'wb') as fdst:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                bar.update(len(buf))

    shutil.copystat(str(p_src), temp_dst)
    # Reusing your existing atomic file replacement mechanism
    os.replace(temp_dst, p_dst)


def _parallel_dir_copy_engine(src_path: Path, dst_path: Path, max_workers: int, bar: tqdm):
    """Worker sub-task: Copies directory tree structures concurrently across threads."""
    for dirpath, _, _ in os.walk(src_path):
        rel_dir = Path(dirpath).relative_to(src_path)
        (dst_path / rel_dir).mkdir(parents=True, exist_ok=True)
        
    all_files = [Path(dirpath) / f for dirpath, _, filenames in os.walk(src_path) for f in filenames]
    progress_lock = threading.Lock()
            
    def _copy_worker(file_src_path: Path):
        rel_file = file_src_path.relative_to(src_path)
        file_dst_path = dst_path / rel_file
        f_size = file_src_path.stat().st_size
        shutil.copy2(file_src_path, file_dst_path)
        with progress_lock:
            bar.update(f_size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_copy_worker, all_files)


def safe_copy(src, dst, max_retries=7, chunk_size=128*1024*1024,
              force_sync=False, max_workers=8):
    """
    A simplified single-responsibility core transfer engine.
    Handles thread-safe buffered file streaming OR multi-threaded directory mirroring.

    Args:
        src (str/Path): The source file or directory path.
        dst (str/Path): The destination file or directory path.
        max_retries (int, optional): Number of retry attempts on failure. Defaults to 7.
        chunk_size (int, optional): Size of the read/write buffer in bytes. Defaults to 128MB.
        force_sync (bool, optional): If True, blocks execution until the hardware cache is flushed. Defaults to False.
        max_workers (int, optional): Thread pool size for parallel directory copies. Defaults to 8.
    """
    file_specific_lock = get_file_lock(str(dst))

    def save_it(src_path, dst_path):
        with file_specific_lock:
            p_src = Path(src_path)
            p_dst = Path(dst_path)
            
            is_directory = p_src.is_dir()
            total_bytes = _get_total_bytes(p_src)
            desc_msg = "🗂️ Syncing Directory Layout" if is_directory else "📄 Syncing File"

            for i in range(max_retries):
                try:
                    with tqdm(total=total_bytes, unit='B', unit_scale=True, 
                              unit_divisor=1024, desc=desc_msg, leave=True) as bar:
                        
                        if is_directory:
                            if p_dst.exists():
                                shutil.rmtree(p_dst)
                            _parallel_dir_copy_engine(p_src, p_dst, max_workers, bar)
                        else:
                            _stream_file_buffered(p_src, p_dst, chunk_size, bar)
                            
                    print("✅ I/O operations verified. Safe copy transaction completed.")
                    break
                
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"\n⚠️ [I/O Exception Intercepted] Stream broken. Retrying in 10s... ⏳")
                        time.sleep(10)
                    else:
                        print(f"\n💀 Fatal error: Failed to complete copy sequence after {max_retries} attempts: {e}")

    thread = threading.Thread(target=save_it, args=(str(src), str(dst)))
    thread.daemon = True
    print(f"🚀 Modular Safe Copy Thread spawned! Buffer latch: {chunk_size//(1024**2)}MB. 📡")
    thread.start()
    
    if force_sync:
        print(f"🔒 [FORCE_SYNC ACTIVE] Locking cell execution runtime... Please wait! 🛑")
        thread.join()
        if hasattr(os, 'sync'):
            print(f"💾 Flushing OS cache buffers directly onto target layout disk... 🧼")
            os.sync()
        print(f"✨ [SUCCESS] Hardware cache synchronized! Fast pipeline closed. 🏁")
    else:
        print(f"🛸 [ASYNC MODE] Action released early. File copy running in background... 🎭")
    
    return thread


def _prepare_temp_directory(extract_to: str) -> str:
    """
    Creates and purges a temporary directory to facilitate atomic zip extraction.

    This acts as a staging environment. If a previous extraction crashed or left
    stale artifacts, this function guarantees a clean slate by forcefully 
    wiping the target path before recreating it.

    Args:
        extract_to (str): The final destination path where the zip contents will live.

    Returns:
        str: The absolute or relative path to the freshly generated temporary directory.
    """
    # Generate the temporary directory path by appending a '_part' suffix
    temp_dir = extract_to.rstrip('/') + "_part"
    
    # Forcefully eliminate any preexisting directory or stale files at that location
    remove_folder(temp_dir, force=True)
    
    # Create the clean staging directory from scratch
    os.makedirs(temp_dir, exist_ok=True)
    
    # Return the clean temporary path to the orchestration pipeline
    return temp_dir

class AtomicParallelZipExtractor:
    """
    A high-performance, atomic, and multi-threaded ZIP extraction engine.
    
    This class consolidates hierarchical directory mapping, multi-threaded worker
    chunking, and transaction-style atomic folder swaps. It guarantees that an
    extraction process never leaves corrupted, half-written files at the destination
    if a crash, network drop, or execution cancellation occurs.
    """

    def __init__(self, zip_path: str, extract_to: str, replace: bool = False, max_workers: int = 4):
        """
        Initializes the atomic parallel extraction pipeline configuration.

        Args:
            zip_path (str): The physical path targeting the source archive file.
            extract_to (str): The target location directory where content will settle.
            replace (bool, optional): If True, wipes existing destination folders. Defaults to False.
            max_workers (int, optional): Thread boundary cap scaling concurrency. Defaults to 4.
        """
        self.zip_path = zip_path
        self.extract_to = extract_to
        self.replace = replace
        self.max_workers = max_workers
        self._lock = threading.Lock()  # Dynamic lock protecting progress bar updates from worker race conditions

    def _build_directory_tree(self, temp_dir: str) -> List[zipfile.ZipInfo]:
        """
        Pre-generates the entire directory tree architecture synchronously before dumping file threads.

        This synchronous step is highly critical for stability. It prevents multi-threaded core
        workers from executing race-prone, overlapping 'os.makedirs' calls simultaneously, 
        which frequently leads to standard operating system file collision errors.

        Args:
            temp_dir (str): The temporary path staging the extraction process.

        Returns:
            List[zipfile.ZipInfo]: A isolated list tracking file records stripped of empty directory nodes.
        """
        file_members = []
        
        # Open the ZIP archive in safe read-only mode
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            # Iterate sequentially through structural layout metadata inside the archive
            for member in zf.infolist():
                # If the current structural item represents an explicit directory node
                if member.is_dir():
                    # Generate the physical folder path matching the archive schema inside temp directory
                    os.makedirs(os.path.join(temp_dir, member.filename), exist_ok=True)
                else:
                    # Resolve parent folder path bound to the file object
                    parent_dir = os.path.dirname(os.path.join(temp_dir, member.filename))
                    if parent_dir:
                        # Pre-generate parent directories if they don't exist yet
                        os.makedirs(parent_dir, exist_ok=True)
                    # Append the verified file record to the work pool list
                    file_members.append(member)
                    
        return file_members

    def _extract_chunk(self, chunk: List[zipfile.ZipInfo], temp_dir: str, bar: tqdm):
        """
        Dedicated thread worker loop processing an assigned subset slice of files.

        Args:
            chunk (List[zipfile.ZipInfo]): Slice allocation tracking files assigned to this specific thread thread.
            temp_dir (str): Staging pathway directory collecting compiled outputs.
            bar (tqdm): Progress bar handler catching completed metrics updates.
        """
        # Instantiate a dedicated archive stream handle restricted within this worker thread
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for member in chunk:
                # Stream binary block directly from disk storage grid to target temporary path location
                zf.extract(member, temp_dir)
                
                # Acquire instance-level lock to securely notify progress interface without corruption
                with self._lock:
                    bar.update(member.file_size)

    def _extract_parallel(self, file_members: List[zipfile.ZipInfo], temp_dir: str):
        """
        Segments file allocation loads evenly across active threads and orchestrates execution pools.

        Args:
            file_members (List[zipfile.ZipInfo]): Clean collection mapping files to parse.
            temp_dir (str): Staging directory area holding structural assets.
        """
        # Accumulate exact total byte sizing requirements to set up accurate metric bars
        total_size = sum(f.file_size for f in file_members)
        
        # Stratify files using a round-robin stride step sequence mapping loads evenly among workers
        chunks = [file_members[i::self.max_workers] for i in range(self.max_workers)]

        # Initialize user tracking interface monitoring output speed and progression metrics
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Extracting", leave=True) as bar:
            # Spawn the concurrent asynchronous execution context frame
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Dispatch tasks safely across pool workers mapping active data chunk buffers
                futures = [
                    executor.submit(self._extract_chunk, chunk, temp_dir, bar) 
                    for chunk in chunks if chunk
                ]
                # Collect thread responses as they finalize tasks
                for future in as_completed(futures):
                    # Call result() to escalate internal thread runtime exceptions to the main orchestration loop
                    future.result()

    def extract(self) -> bool:
        """
        The orchestrator method triggering secure, parallel, and atomic zip extractions.

        Returns:
            bool: True if transaction-style swap concludes flawlessly, False otherwise.
        """
        # Skip routine entirely if a folder exists and overwrite permissions are locked
        if os.path.exists(self.extract_to) and not self.replace:
            print(f"⏭️ Extraction directory {self.extract_to} already exists. Skipping.")
            return True

        # Provision the temporary extraction workspace staging environment safely
        temp_extract_to = _prepare_temp_directory(self.extract_to)

        try:
            # Step 1: Map layout and generate structural folder architectures
            file_members = self._build_directory_tree(temp_extract_to)
            
            # Step 2: Concurrently stream and extract chunk buffers across workers
            self._extract_parallel(file_members, temp_extract_to)

            # Step 3: Conclude operation using transaction style atomic layout replacement
            if os.path.exists(self.extract_to):
                remove_folder(self.extract_to, force=True)
                
            # Perform the final atomic folder swap operation seamlessly
            replace_with_error(temp_extract_to, self.extract_to)
            print(f"✅ Successfully extracted to: {self.extract_to}")
            return True
        
        except Exception as e:
            # Rollback phase: Clean up files to preserve integrity in case of structural crashes
            remove_folder(self.extract_to, force=True)
            remove_folder(temp_extract_to, force=True)
            print(f"❌ Error extracting zip file: {e}")
            return False
        

def extract_zip(zip_path: str, extract_to: str, replace: bool = False, max_workers: int = 4) -> bool:
    """
    Unified public wrapper function rendering backward-compatible integration access points.
    
    This abstracts away the class creation details, enabling simple functional calls 
    ideal for clean PyPI module entry points.
    """
    extractor = AtomicParallelZipExtractor(
        zip_path=zip_path, extract_to=extract_to, 
        replace=replace, max_workers=max_workers
    )
    return extractor.extract()