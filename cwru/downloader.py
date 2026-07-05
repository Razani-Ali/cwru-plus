# src/downloader.py

"""
CWRU Dataset Downloader Module

This module is responsible for orchestrating the download process of the Case Western Reserve 
University (CWRU) bearing dataset. It parses configuration files for URLs, constructs 
standardized filenames, and delegates the heavy lifting to the concurrent utility downloader.
"""

import os
from typing import Dict, Any

# Importing the generalized parallel execution engine from the local utilities module
from .utilities import run_parallel_downloads

# Attempt to import dataset URL configurations safely
try:
    from config.urls48 import CWRU_48kHz_links
    from config.urls12 import CWRU_12kHz_links
except ImportError:
    # Ignore import errors silently; the user might handle them elsewhere or provide alternatives
    pass

def generate_cwru_filename(group_name: str, file_info: Dict[str, Any]) -> str:
    """
    Constructs a standardized, uniform filename for a CWRU dataset file based on its metadata.
    
    Args:
        group_name (str): The primary category of the fault (e.g., 'Normal', 'Ball', 'IR', 'OR').
        file_info (Dict[str, Any]): A dictionary containing metadata such as horsepower ('HP'), 
                                    fault diameter ('diameter'), and sensor location ('location').
                                    
    Returns:
        str: The fully formatted string filename ending with '.mat'.
    """
    # Extract the horsepower load value from the file metadata dictionary
    hp = file_info.get('HP')
    
    # If the data group is 'Normal' (baseline), it has no fault diameter or location
    if group_name == 'Normal':
        # Return a simple filename formatted for normal baseline data
        return f"{group_name}_{hp}HP.mat"
        
    # Extract the fault diameter (severity) from the file metadata
    diameter = file_info.get('diameter')
    # Extract the sensor location (e.g., 3, 6, 12 o'clock) from the file metadata
    location = file_info.get('location')
    
    # Construct the base filename incorporating the fault group, diameter, and horsepower
    filename = f"{group_name}_Diameter{diameter}_{hp}HP"
    
    # If a specific location is provided in the metadata, append it to the filename
    if location is not None:
        filename += f"_Location{location}"
        
    # Append the '.mat' extension to the final string and return
    return filename + ".mat"

def download_CWRU_files(
    download_path: str = 'data/CWRU12/RawFiles',
    CWRUfs: int = 12,
    max_retries: int = 10,
    retry_delay: int = 3,
    replace_files: bool = False,
    max_workers: int = 3
) -> bool:
    """
    Orchestrates the downloading of the CWRU dataset files using parallel execution.
    
    Args:
        download_path (str, optional): The target directory to save the raw files. Defaults to 'data/CWRU12/RawFiles'.
        CWRUfs (int, optional): The sampling frequency to download (12 or 48). Defaults to 12.
        max_retries (int, optional): Number of retry attempts per file. Defaults to 10.
        retry_delay (int, optional): Seconds to wait before a retry. Defaults to 3.
        replace_files (bool, optional): If True, overwrites existing files. Defaults to False.
        max_workers (int, optional): Number of parallel download threads. Defaults to 3.
        
    Returns:
        bool: True if all files were downloaded (or already existed) successfully, False if any failed.
        
    Raises:
        ValueError: If an unsupported sampling frequency is requested.
    """
    # Ensure the destination directory exists; create it if it doesn't
    os.makedirs(download_path, exist_ok=True)

    # Select the appropriate URL dictionary based on the requested sampling frequency
    if CWRUfs == 12:
        # Load 12 kHz dataset links
        all_groups = CWRU_12kHz_links()
    elif CWRUfs == 48:
        # Load 48 kHz dataset links
        all_groups = CWRU_48kHz_links()
    else:
        # Raise an error if the user requests an unsupported sampling frequency
        raise ValueError(f"Dataset with sampling frequency {CWRUfs} is not supported.")

    # 1. Build the task list using the single-responsibility filename generator
    tasks = []
    # Iterate over each category/group in the configured links
    for group_name, files in all_groups.items():
        # Iterate over each individual file's metadata dictionary within the group
        for f_info in files:
            # Generate the target filename using the helper function
            filename = generate_cwru_filename(group_name, f_info)
            # Combine the download directory path with the generated filename
            output_path = os.path.join(download_path, filename)
            # Append the task tuple (URL, target_path, filename) to the execution list
            tasks.append((f_info['url'], output_path, filename))

    # Calculate the total number of files queued for download
    total_files = len(tasks)
    # Notify the user that the parallel downloading process is starting
    print(f"▶️ Starting parallel download of {total_files} files using {max_workers} workers...")

    # 2. Delegate to the core parallel downloader
    # Execute the downloads concurrently and capture the success count and list of failures
    successful, failed_downloads = run_parallel_downloads(
        tasks=tasks, 
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        replace_files=replace_files
    )

    # 3. Present the summary
    # Print a formatted summary of the download operation
    print("\nDownload summary:")
    print(f"❇️  Total files attempted: {total_files}")
    print(f"✅  Successful downloads: {successful}")
    print(f"❌  Failed downloads: {len(failed_downloads)}")

    # Check if there were any files that failed to download
    if failed_downloads:
        # Print the list of failed files for user visibility
        print("\nThe following files could not be downloaded:")
        for f in failed_downloads:
            print("   -", f)
    else:
        # Confirm full success if the failed list is empty
        print("All files downloaded successfully!")

    # Print the final storage location
    print(f"Files saved to: {download_path}\n")

    # Return True only if the number of successful downloads matches the total requested files
    return total_files == successful