"""
End-to-End Pipeline Execution Script

This script serves as a complete testing and execution pipeline for the CWRU-Plus dataset.
It covers concurrent downloading, automated zipping, background safe copying to persistent 
storage (e.g., Google Drive), and parallel data ingestion.
"""

import numpy as np
# Since the cwru module is directly attached to the namespace,
# the user can import it easily without needing complex prefixes.
import cwru
from cwru.utilities import create_zip, safe_copy

def main():
    """
    Main orchestrator function for the CWRU dataset pipeline.
    Executes the downloading, archiving, backing up, and ingestion phases sequentially.
    """
    print("==================================================")
    print("🚀 Starting CWRU-Plus End-to-End Pipeline Test")
    print("==================================================\n")

    # Set default paths for data storage (Relative to the project root)
    RAW_DIR = "data/CWRU12/RawFiles"
    INGESTED_NPZ = "data/CWRU12/CWRU12Ingested.npz"
    # FILTERED_NPZ = "data/CWRU12/CWRU12Filtered.npz"

    # Define prefixes for cloud/Colab environments
    # temp_prefix specifies local ephemeral storage for fast I/O
    temp_prefix = "/content/"
    # permanent_prefix specifies mounted persistent storage (e.g., Google Drive)
    permanent_prefix = "/content/drive/MyDrive/shared_folder/"

    # Construct the full temporary (fast) paths
    RAW_DIR_temp = temp_prefix + RAW_DIR
    INGESTED_NPZ_temp = temp_prefix + INGESTED_NPZ
    # FILTERED_NPZ_temp = temp_prefix + FILTERED_NPZ

    # Construct the full permanent (backup) paths
    RAW_DIR_perm = permanent_prefix + RAW_DIR
    # INGESTED_NPZ_perm = permanent_prefix + INGESTED_NPZ
    # FILTERED_NPZ_perm = permanent_prefix + FILTERED_NPZ


    # ----------------------------------------------------------------
    # Step 1: Download Data (Concurrent Downloading)
    # ----------------------------------------------------------------
    print("--- [STEP 1] Downloading Raw Files ---")
    # We select a 12 kHz sampling frequency for this test
    download_success = cwru.download(
        CWRUfs=12,
        download_path=RAW_DIR_temp,
        replace_files=False,
        max_workers=8
    )
    # Halt execution if the download phase fails
    if not download_success:
        print("❌ Download phase failed. Exiting.")
        return
    
    # Archive the downloaded raw files into a single ZIP file for easier management
    ok = create_zip(
        source_dir = RAW_DIR_temp,
        zip_path = RAW_DIR_temp + ".zip", 
        replace = False
    )
    
    # Spawn a background daemon thread to safely copy the ZIP archive to persistent storage
    thrd = safe_copy(
        src = RAW_DIR_temp + ".zip", 
        dst = RAW_DIR_perm + ".zip"
    )


    # ----------------------------------------------------------------
    # Step 2: Parallel Data Ingestion
    # ----------------------------------------------------------------
    print("\n--- [STEP 2] Ingesting .MAT Files into Unified .NPZ ---")
    # Convert all raw .mat files into a unified, aligned .npz database
    ingest_success = cwru.ingest(
        base_path=RAW_DIR_temp,
        # The internal method automatically appends the .npz extension
        output_file_path=INGESTED_NPZ_temp.replace(".npz", ""), 
        replace_file=True,
        dtype=np.float32
    )
    # Halt execution if the ingestion phase fails
    if not ingest_success:
        print("❌ Ingestion phase failed. Exiting.")
        return
    
    # Spawn a background daemon thread to back up the ingested dataset to persistent storage
    thrd = safe_copy(src = RAW_DIR_temp, dst = RAW_DIR_perm)


    # ----------------------------------------------------------------
    # Step 3: Apply Custom Digital or Non-linear Filters (Commented Out)
    # ----------------------------------------------------------------
    print("\n--- [STEP 3] Signal Filtering (Skipped / Commented Out) ---")
    """
    import scipy.signal as sig

    # Define specific filters for each channel (Demonstrating Inversion of Control)
    def drive_end_filter(signal):
        # 4th-order low-pass Butterworth filter with zero phase shift (filtfilt)
        b, a = sig.butter(4, 0.1, btype='low')
        return sig.filtfilt(b, a, signal)

    def fan_end_filter(signal):
        # Non-linear median filter with a kernel size of 5 to remove impulsive noise
        return sig.medfilt(signal, kernel_size=5)

    custom_filter_dict = {
        "DE": drive_end_filter,
        "FE": fan_end_filter,
        "BA": None  # Keep the base accelerometer channel raw (unfiltered)
    }

    print("Running parallel filtering on dataset...")
    cwru.filter(
        filters=custom_filter_dict,
        input_npz=INGESTED_NPZ_temp,
        output_npz=FILTERED_NPZ_temp,
        max_workers=8
    )
    
    # If you activate the filter, set the TARGET_PATH variable to FILTERED_NPZ in Step 4.
    
    # Safely back up the filtered dataset to persistent storage
    thrd = safe_copy(src = FILTERED_NPZ_temp, dst = FILTERED_NPZ_perm)
    """
    
    print("🎉 End-to-End Pipeline Executed Successfully!")

if __name__ == "__main__":
    main()