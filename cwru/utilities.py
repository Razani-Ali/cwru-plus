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
from typing import List, Tuple, Dict, Any
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

def create_zip(source_dir: str, zip_path: str, replace: bool = False):
    """
    Recursively archives a directory into a ZIP file with a progress bar.
    
    Args:
        source_dir (str): The root directory to be archived.
        zip_path (str): The destination path for the final .zip file.
        replace (bool, optional): Whether to overwrite an existing ZIP file. Defaults to False.
        
    Returns:
        bool: True if the zip archive is successfully created, False otherwise.
    """
    # Check if the zip file already exists and should be preserved
    if os.path.exists(zip_path) and not replace:
        print(f"⏭️ Zip file {zip_path} already exists. Skipping.")
        return True
    
    # Initialize a list to hold all target file paths and a counter for total byte size
    file_tasks = []
    total_size = 0

    # Define a recursive function to map all files in the source directory tree
    def collect_files(target_dir):
        nonlocal total_size
        # Scan the directory efficiently using os.scandir
        with os.scandir(target_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    # If it's a file, add its path to the task list and aggregate its size
                    file_tasks.append(entry.path)
                    total_size += entry.stat().st_size
                elif entry.is_dir():
                    # If it's a directory, recursively explore it
                    collect_files(entry.path)

    # Populate the file_tasks list and calculate total_size
    collect_files(source_dir)

    # Ensure the parent directory for the output zip exists
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    # Generate a temporary path for the zip to prevent corrupted partial archives
    temp_zip = get_temp_path(zip_path)

    try:
        # Open the temporary zip file in write mode using standard DEFLATED compression
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Initialize a tqdm progress bar tracking bytes
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
                      desc="Zipping", leave=True) as bar:
                
                # Iterate through all collected file paths
                for file_path in file_tasks:
                    # Calculate the relative path to maintain folder structure inside the zip
                    rel_path = os.path.relpath(file_path, source_dir)
                    # Write the file into the zip archive
                    zipf.write(file_path, rel_path)
                    # Update the progress bar by the exact size of the processed file
                    bar.update(os.path.getsize(file_path))
        
        # Atomically replace the temporary zip with the final destination path
        replace_with_error(temp_zip, zip_path)
        print(f"✅ Successfully created zip at: {zip_path}")
        return True
    except Exception as e:
        # On failure, log the error and clean up the temporary zip file
        print(f"❌ Error creating zip file: {e}")
        remove_file(temp_zip, force=True)
        return False
    
# Dictionary to store thread locks mapped to specific file paths
_locks = {}
# A master lock to prevent race conditions when creating new file-specific locks
_master_lock = threading.Lock()

def get_file_lock(file_path):
    """
    Retrieves or creates a thread lock specific to a file path.
    
    Args:
        file_path (str): The unique string representation of the target file path.
        
    Returns:
        threading.Lock: A lock object assigned exclusively to the requested file path.
    """
    # Acquire the global master lock to safely evaluate the _locks dictionary
    with _master_lock:
        # If the file path doesn't have a lock yet, instantiate one
        if file_path not in _locks:
            _locks[file_path] = threading.Lock()
        # Return the specific lock for this file
        return _locks[file_path]

def safe_copy(src, dst, max_retries=7, chunk_size=50*1024*1024):
    """
    Spawns a daemon thread to copy a file safely, atomically, and with thread locks.
    
    Args:
        src (str/Path): The source file path.
        dst (str/Path): The destination file path.
        max_retries (int, optional): Number of retry attempts on failure. Defaults to 7.
        chunk_size (int, optional): Size of the read/write buffer in bytes (default 50MB).
        
    Returns:
        threading.Thread: The daemon thread handling the copy operation.
    """
    # Retrieve the thread lock dedicated to the destination file
    file_specific_lock = get_file_lock(str(dst))
    
    # Define the internal function to be executed by the thread
    def save_it(src_path, dst_path):
        # Acquire the file-specific lock to ensure only one thread modifies this destination
        with file_specific_lock:
            # Initiate the retry loop
            for i in range(max_retries):
                try:
                    # Convert paths to pathlib objects
                    p_src = Path(src_path)
                    p_dst = Path(dst_path)
                    
                    # Ensure the destination directory exists
                    p_dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Create a temporary destination path to prevent file corruption
                    temp_dst = p_dst.with_suffix('.tmp')
                    
                    # Open source for reading and temp file for writing (both in binary mode)
                    with open(p_src, 'rb') as fsrc:
                        with open(temp_dst, 'wb') as fdst:
                            while True:
                                # Read data in chunks
                                buf = fsrc.read(chunk_size)
                                # Break the loop if the end of the file is reached
                                if not buf:
                                    break
                                # Write the chunk to the temporary file
                                fdst.write(buf)

                    # Preserve the original file metadata (permissions, timestamps, etc.)
                    shutil.copystat(src_path, temp_dst)
                    
                    # Atomically replace the temp file with the final destination
                    replace_with_error(temp_dst, p_dst)
                    
                    # Break the retry loop upon success
                    break
                
                except Exception as e:
                    # If not on the last attempt, wait 20 seconds before retrying
                    if i < max_retries - 1:
                        time.sleep(20)
                    else:
                        # Log failure if all retries are exhausted
                        print(f"❌ Failed to copy after {max_retries} attempts: {e}")

    # Instantiate a new thread targeting the internal save_it function
    thread = threading.Thread(
        target=save_it, 
        args=(str(src), str(dst))
    )
    # Set the thread as a daemon so it doesn't block the main program from exiting
    thread.daemon = True
    # Start the thread execution
    thread.start()
    
    # Return the thread object in case the caller needs to join() or monitor it
    return thread