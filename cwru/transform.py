"""
Module: transform.py
Description: An advanced dimension-agnostic feature extraction framework.
Supports both standard 3D signal tensors (Windows, Channels, Length) and 
4D multi-fold tensors (Folds, Windows, Channels, Length). Utilizes fast memory views 
to unify iteration loops while preserving the strict structural geometry upon storage.
"""

import os
from typing import Callable, Tuple, Any
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class FeatureWorkspace:
    """
    Manages offline feature extraction and automated structural tensor reshaping.
    Dynamically adapts to the dimensionality of input signals without memory copying.
    """

    @staticmethod
    def transform_and_save(
        X: np.ndarray,
        y: np.ndarray,
        file_ids: np.ndarray,
        metadata: Any,
        transform_fn: Callable[[np.ndarray], np.ndarray],
        save_path: str
    ) -> None:
        """
        Processes 3D or 4D signal tensors by utilizing a flattened temporary view 
        for uniform execution, then automatically reconstructs and archives the original 
        dimensional layouts.

        Args:
            X (np.ndarray): Input tensor. Can be 3D (Windows, Channels, Length) 
                            or 4D (Folds, Windows, Channels, Length).
            y (np.ndarray): Target class labels.
            file_ids (np.ndarray): Origin tracking array to prevent downstream leakage.
            transform_fn (Callable): Injected extractor function. Accepts a 1D/2D channel window.
            save_path (str): Destination disk path for the compressed `.npz` container.
        """
        input_ndim = X.ndim
        
        if input_ndim == 3:
            # Standard 3D structural mapping
            total_windows, num_channels, signal_len = X.shape
            num_folds = 1
            X_view = X # No flattening needed for 3D tensors
            print(f"📦 Detected 3D Signal Tensor: {total_windows} windows, {num_channels} channels.")
            
        elif input_ndim == 4:
            # Advanced 4D Folded structural mapping
            num_folds, total_windows, num_channels, signal_len = X.shape
            print(f"🔄 Detected 4D Folded Tensor: {num_folds} Folds, {total_windows} Windows per fold.")
            
            # Create a zero-copy memory view merging Folds and Windows into a unified dimension
            # Combined Shape: (Folds * Windows, Channels, Length)
            X_view = X.reshape(num_folds * total_windows, num_channels, signal_len)
        else:
            raise ValueError(f"Unsupported tensor dimension: {input_ndim}D. Expected 3D or 4D array.")

        total_iterations = X_view.shape[0]
        extracted_features_list = []
        
        # Setup the progress bar context
        iterator = range(total_iterations)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Extracting Feature Space", unit="step")
            
        # Execute the extraction across the optimized sequential memory view
        for i in iterator:
            single_window = X_view[i] # Shape: (Channels, Length)
            feature_vector = transform_fn(single_window)
            extracted_features_list = [] if feature_vector is None else extracted_features_list
            extracted_features_list.append(feature_vector)
            
        # Determine individual feature shape returned by the callable
        sample_feature = np.array(extracted_features_list[0])
        feature_dimensions = sample_feature.shape # Captures whatever dimensions the user creates
        
        # Stack the processed samples along the combined virtual batch dimension
        flat_feature_matrix = np.stack(extracted_features_list, axis=0)
        
        # Reconstruct original structural dimensions securely based on the input state
        if input_ndim == 3:
            # Shape mapping: (Windows, Extracted_Features...)
            final_feature_tensor = flat_feature_matrix
        else:
            # Shape mapping: (Folds, Windows, Extracted_Features...)
            # We restore 'Folds' and 'Windows' to their pristine separate dimensions
            reconstruct_shape = (num_folds, total_windows) + feature_dimensions
            final_feature_tensor = flat_feature_matrix.reshape(reconstruct_shape)
            
        # Secure system directories
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        # Compress and archive the mathematically intact features alongside standard trackers
        np.savez_compressed(
            save_path,
            features=final_feature_tensor,
            labels=y,
            file_ids=file_ids,
            metadata=metadata
        )
        
        print(f"✅ Extraction complete! Tensor saved successfully at: '{save_path}'")
        print(f"📊 Final Saved Feature Shape: {final_feature_tensor.shape}")

    @staticmethod
    def load(npz_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Any]:
        """Restores the pristine multi-dimensional dataset archive from disk."""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Feature archive missing at path: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            features = data['features']
            labels = data['labels']
            file_ids = data['file_ids']
            metadata_raw = data['metadata']
            metadata = metadata_raw.item() if metadata_raw.ndim == 0 else metadata_raw

        return (features, labels, file_ids), metadata