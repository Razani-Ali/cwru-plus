import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List, Optional, Dict


class CWRURecordFilter:
    """
    Encapsulates the logic for filtering dataset records based on a specified strategy.
    
    This class evaluates raw CWRU dataset metadata (fault type, severity, and location)
    to determine which signal records should be included in the final dataset for
    condition monitoring tasks.
    """
    
    def __init__(self, strategy: str = "standard"):
        """
        Initializes the filter with a specific inclusion strategy.
        
        Args:
            strategy (str): The filtering strategy to apply. Options typically include
                            'standard', 'extended', or 'full'.
        """
        self.strategy = strategy

    def is_valid(self, fault: str, severity: str, loc: int) -> bool:
        """
        Evaluates a single record's metadata against the filtering strategy.
        
        Args:
            fault (str): The fault category (e.g., 'Normal', 'Ball', 'IR', 'OR').
            severity (str): The fault diameter/severity (e.g., '007', '014').
            loc (int): The sensor location or fault position (e.g., 6, 3, 12, or -1 for none).
            
        Returns:
            bool: True if the record meets the criteria of the selected strategy, False otherwise.
        
        Raises:
            ValueError: If an unknown strategy name is provided.
        """
        # If the strategy is 'full', bypass all checks and include everything
        if self.strategy == "full":
            return True
            
        # Define baseline conditions for valid records
        is_normal = (fault == 'Normal')
        valid_sev = severity in ['', '007', '014', '021']
        
        # Note: Validates specific standard locations (e.g., 6 o'clock or baseline -1)
        valid_loc = (loc == 6) or (loc == -1)
        
        # Drive End (DE) faults usually include Ball, Inner Race (IR), and Outer Race (OR)
        is_de_fault = fault in ['Ball', 'IR', 'OR'] and valid_sev and valid_loc
        # Fan End (FE) faults
        is_fe_fault = fault in ['FE_B', 'FE_IR', 'FE_OR'] and valid_sev and valid_loc
        
        # Apply the logic based on the chosen strategy
        if self.strategy == "standard":
            return is_normal or is_de_fault
        elif self.strategy == "extended":
            return is_normal or is_de_fault or is_fe_fault
            
        raise ValueError(f"🚫 Unknown filter strategy: {self.strategy}")

    def filter_indices(self, data: Dict[str, np.ndarray]) -> List[int]:
        """
        Iterates through the entire dataset and extracts the indices of valid records.
        
        Args:
            data (Dict[str, np.ndarray]): The materialized dictionary containing dataset arrays.
            
        Returns:
            List[int]: A list of row indices that pass the filtering logic.
        """
        return [
            i for i in range(len(data["Fault"]))
            if self.is_valid(data["Fault"][i], data["Severity"][i], data["Location"][i])
        ]


class CWRULabelGenerator:
    """
    Generates formatted textual labels based on user preference for classification tasks.
    """
    
    @staticmethod
    def generate(y: np.ndarray, s: np.ndarray, loc: np.ndarray, category: str) -> np.ndarray:
        """
        Constructs target label arrays by combining different metadata fields.
        
        Args:
            y (np.ndarray): Array of fault types.
            s (np.ndarray): Array of fault severities.
            loc (np.ndarray): Array of fault locations.
            category (str): The desired label format ('only types', 'types & severity', 
                            or 'types, severity & locations').
                            
        Returns:
            np.ndarray: An array of formatted string labels corresponding to each sample.
            
        Raises:
            ValueError: If an unsupported category is provided.
        """
        # Return base fault types only (e.g., 'IR', 'Normal')
        if category == 'only types':
            return np.array(y, dtype=str)
        # Concatenate type and severity (e.g., 'IR007')
        elif category == 'types & severity':
            return np.array([f"{typ}{sev}" for typ, sev in zip(y, s)], dtype=str)
        # Concatenate type, severity, and location using an '@' separator (e.g., 'OR014@6')
        elif category == 'types, severity & locations':
            return np.array([
                f"{typ}{sev}{'@' + str(l) if l != -1 else ''}"
                for typ, sev, l in zip(y, s, loc)
            ], dtype=str)
        
        raise ValueError("🚫 Invalid fault_category provided.")


class WindowingEngine:
    """
    Handles high-performance sliding window extraction utilizing zero-copy memory mapping.
    """
    
    def __init__(self, window_size: int, step_size: int):
        """
        Initializes the windowing parameters.
        
        Args:
            window_size (int): The number of data points in a single observation window.
            step_size (int): The stride/overlap step between consecutive windows.
        """
        self.window_size = window_size
        self.step_size = step_size

    def apply(self, signal_chunk: np.ndarray) -> np.ndarray:
        """
        Applies a sliding window over the time axis of a signal matrix.
        
        Utilizes NumPy's stride tricks to extract overlapping windows instantaneously 
        without copying data in memory.
        
        Args:
            signal_chunk (np.ndarray): A 2D array of shape (Channels, Sequence_Length).
            
        Returns:
            np.ndarray: A 3D tensor of shape (Num_Windows, Channels, Window_Size),
                        which is directly compatible with 1D CNNs.
        """
        # Extract sliding windows along the time axis (axis=1)
        view = sliding_window_view(signal_chunk, window_shape=self.window_size, axis=1)
        # Apply the step/stride to reduce the number of overlapping windows
        view = view[:, ::self.step_size, :]
        # Transpose dimensions to match deep learning expected formats: (Batch, Channel, Length)
        return view.transpose(1, 0, 2)


class TemporalPartitioner:
    """
    Splits a full continuous signal matrix into equal, non-overlapping temporal chunks.
    Useful for creating time-based cross-validation folds to prevent data leakage.
    """
    
    def __init__(self, parts_count: int, min_length: int):
        """
        Initializes the partitioner.
        
        Args:
            parts_count (int): The number of chunks to divide the signal into.
            min_length (int): The minimum acceptable length for a chunk (usually window_size).
        """
        self.parts_count = parts_count
        self.min_length = min_length

    def partition(self, signal_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Slices the original signal matrix into temporally ordered segments.
        
        Args:
            signal_matrix (np.ndarray): 2D array of continuous signals (Channels, Total_Length).
            
        Returns:
            List[np.ndarray]: A list containing the segmented signal arrays. Returns an empty 
                              list if the resulting chunks would be smaller than the minimum length.
        """
        total_length = signal_matrix.shape[1]
        part_length = total_length // self.parts_count
        
        # Security guard: ensures chunks are large enough to be windowed later
        if part_length < self.min_length:
            return []
            
        # Use list comprehension to dynamically slice the array along the time axis for all channels
        return [
            signal_matrix[:, i * part_length : (i + 1) * part_length]
            for i in range(self.parts_count)
        ]


class RecordStorage:
    """
    Manages the memory, aggregation, and concatenation of extracted windows and metadata.
    Employs ultra-fast NumPy vectorization to map file-level metadata to window-level metadata.
    """
    
    def __init__(self):
        """Initializes empty storage lists for signals and corresponding metadata."""
        self.X = []
        self.window_counts = []  # ⭐️ Keeps track of window counts instead of large lists
        self.Y, self.S, self.HP, self.Loc, self.file_idx = [], [], [], [], []

    def add_windows(self, windows: np.ndarray, fault: str, severity: str, hp: int, loc: int, file_idx: int):
        """
        Registers a batch of extracted windows and stores their shared metadata once.
        
        Args:
            windows (np.ndarray): Extracted 3D tensor of windows for a specific record.
            fault (str): Fault type.
            severity (str): Fault severity.
            hp (int): Horsepower load.
            loc (int): Sensor location.
            file_idx (int): File ID
        """
        self.X.append(windows)
        self.window_counts.append(windows.shape[0])
        
        # ⭐️ We only store the scalars once per file/chunk to save memory and processing time
        self.Y.append(fault)
        self.S.append(severity)
        self.HP.append(hp)
        self.Loc.append(loc)
        self.file_idx.append(file_idx)

    def compile(self, fault_category: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts the collected lists into final, flattened NumPy arrays.
        
        Args:
            fault_category (str): The label generation strategy for target labels.
            
        Returns:
            Tuple: A tuple containing the stacked window tensor (X_out) and vectorized
                   metadata arrays (Y_out, S_out, HP_out, Loc_out).
                   
        Raises:
            ValueError: If no windows were added to the storage prior to compilation.
        """
        
        if not self.X:
            raise ValueError("No valid windows were extracted. Check your filter strategy or window size.")
            
        # Concatenate all window batches into a single master tensor
        X_out = np.concatenate(self.X, axis=0).astype(np.float32)
        
        # ⭐️ Ultra-fast vectorization using np.repeat to expand file-level metadata 
        # to match the exact number of extracted windows per file.
        counts = np.array(self.window_counts, dtype=np.int32)
        Y_base = np.repeat(self.Y, counts)
        S_out = np.repeat(self.S, counts)
        HP_out = np.repeat(self.HP, counts).astype(np.int32)
        Loc_out = np.repeat(self.Loc, counts).astype(np.int32)
        File_idx_out = np.repeat(self.file_idx, counts).astype(np.int32)
        
        # Generate dynamic, formatted string labels
        Y_out = CWRULabelGenerator.generate(Y_base, S_out, Loc_out, fault_category)
        
        return X_out, Y_out, S_out, HP_out, Loc_out, File_idx_out


class CWRUDatasetBuilder:
    """
    Master orchestrator class combining filters, windowing, temporal partitioning, 
    and memory storage into a cohesive dataset generation pipeline.
    """
    
    def __init__(self, npz_path: str = 'data/CWRU12/CWRU12Ingested.npz'):
        """
        Initializes the builder with the target database file.
        
        Args:
            npz_path (str): The path to the ingested .npz dataset file.
        """
        self.npz_path = npz_path

    def build(
        self, 
        window_size: int, 
        step_size: int, 
        sensors: List[str] = ["DE", "FE"], 
        strategy: str = "standard", 
        fault_category: str = "types & severity",
        num_parts: Optional[int] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
        """
        Executes the dataset building pipeline.
        
        Loads data into RAM, applies filters, slices signals into partitions (if specified),
        extracts overlapping windows, and formats the output tensors.
        
        Args:
            window_size (int): Size of the sliding window (number of data points).
            step_size (int): Stride for the sliding window.
            sensors (List[str]): List of channels to extract (e.g., ["DE", "FE"]).
            strategy (str): Filtering strategy for the metadata.
            fault_category (str): Label formatting strategy.
            num_parts (Optional[int]): Number of temporal folds to create. If None, flattens the output.
            
        Returns:
            Tuple: A nested tuple structured as:
                   ((X_tensor, Y_labels), (Severities, Horsepowers, Locations, Sensor_Order))
                   Outputs are either 3D (flattened) or 4D (if num_parts is specified).
        """
        
        print(f"📦 Loading preprocessed dataset from {self.npz_path}...")
        raw_data = np.load(self.npz_path, allow_pickle=True)
        
        # ⭐️ Golden Trick: Materialize data into RAM to bypass NumPy's lazy loading bottleneck.
        # This converts the NpzFile object into a standard Python dictionary instantly.
        data = {key: raw_data[key] for key in raw_data.files}
        raw_data.close() 
        
        # 1. Initialize Components (Composition)
        # Determine active partitions (default to 1 if no temporal folding is requested)
        parts_count = num_parts if num_parts is not None else 1
        record_filter = CWRURecordFilter(strategy)
        partitioner = TemporalPartitioner(parts_count, window_size)
        window_engine = WindowingEngine(window_size, step_size)
        
        # Create separate storage instances for each temporal fold/part
        storages = [RecordStorage() for _ in range(parts_count)]
        
        # 2. Filter valid records based on metadata strategy
        valid_indices = record_filter.filter_indices(data)
        print(f"⚙️ Processing {len(valid_indices)} records (Window={window_size}, Parts={parts_count})...")
        
        # 3. Main Processing Loop (Clean and readable structure)
        for idx in valid_indices:
            # Stack selected sensor arrays into a single 2D matrix: (Channels, Sequence_Length)
            signal_matrix = np.stack([data[s][idx] for s in sensors], axis=0)
            
            # Slice the matrix into temporal segments
            chunks = partitioner.partition(signal_matrix)
            
            # Process each segment independently
            for p, chunk in enumerate(chunks):
                # Apply sliding window view mapping
                windows = window_engine.apply(chunk)
                
                # Register the extracted tensor and its file-level metadata to the corresponding part storage
                storages[p].add_windows(
                    windows, 
                    fault=data["Fault"][idx], 
                    severity=data["Severity"][idx], 
                    hp=data["HorsePower"][idx], 
                    loc=data["Location"][idx],
                    file_idx=idx,
                )

        # 4. Compile Outputs
        # Execute the fast vectorization process for each temporal part
        compiled_parts = [storage.compile(fault_category) for storage in storages]
        
        # Transpose list of tuples -> Tuple of lists: ([X1, X2], [Y1, Y2], ...)
        # This groups all features (X), labels (Y), etc., across parts together
        X_list, Y_list, S_list, HP_list, Loc_list, file_idx_list = zip(*compiled_parts)
        
        # 5. Format Final Output
        if num_parts is None:
            # Flattened output: Pull the first index since there is only 1 part
            X_f, Y_f, S_f, HP_f, Loc_f, idx_f = X_list[0], Y_list[0], S_list[0], HP_list[0], Loc_list[0], file_idx_list[0]
        else:
            # Temporal output: Stack lists into an additional 0-th dimension for Cross-Validation folds
            X_f = np.stack(X_list, axis=0)
            Y_f = np.stack(Y_list, axis=0)
            S_f = np.stack(S_list, axis=0)
            HP_f = np.stack(HP_list, axis=0)
            Loc_f = np.stack(Loc_list, axis=0)
            idx_f = np.stack(file_idx_list, axis=0)
            
        print(f"🎉 Dataset built successfully! Tensor Shape: {X_f.shape}")
        
        # ⭐️ Added tuple(sensors) to match the expected return signature for downstream modules
        return (X_f, Y_f, idx_f), (S_f, HP_f, Loc_f)


def generate_stratified_file_split(
    y: np.ndarray, 
    file_ids: np.ndarray, 
    train_ratio: float = 0.75, 
    val_ratio: float = 0.25, 
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates stratified random split indices based on file IDs to prevent data leakage.
    
    This function ensures that:
    1. Windows from the same physical file are never split across Train/Val/Test sets.
    2. The class distribution (stratification) is maintained across the splits.

    Args:
        y (np.ndarray): 1D array of labels for each window.
        file_ids (np.ndarray): 1D array of file identifiers corresponding to each window.
        train_ratio (float): Proportion of files to include in the train split.
        val_ratio (float): Proportion of files to include in the validation split.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (train_indices, val_indices, test_indices)
    """

    if (train_ratio + val_ratio) > 1.0:
        raise ValueError(f"Sum of ratios must be less or equal to 1.0, got {train_ratio + val_ratio}")

    np.random.seed(random_seed)
    
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        # 1. Find Related Indices
        label_mask = (y == label)
        class_indices = np.where(label_mask)[0]
        
        # 2. Extract and Shuffle Unique Files
        files_in_label = np.unique(file_ids[label_mask])
        np.random.shuffle(files_in_label)
        
        total_files = len(files_in_label)
        if total_files == 0:
            continue
            
        # 3. Slicing calculations for very low file counts (e.g., CWRU/MAFAULDA)
        test_ratio = 1.0 - (train_ratio + val_ratio)
        
        if total_files >= 4 and test_ratio > 0.02 and val_ratio > 0.02:
            # Standard balanced split for adequate file sizes
            train_count = int(np.floor(total_files * train_ratio))
            val_count = int(np.ceil(total_files * val_ratio))
            
            if train_count + val_count >= total_files:
                train_count = max(1, total_files - val_count - 1)
            
            train_end = max(1, train_count)
            val_end = train_end + max(1, val_count)
            
        else:
            # Handle extreme low-file regimes (e.g., exactly 4 files per class)
            if test_ratio <= 0.02 or np.isclose(test_ratio, 0.0):
                # 2-way split (Train / Val only, No Test set)
                if total_files == 4:
                    train_end = 3
                    val_end = 4
                else:
                    train_end = max(1, int(np.round(total_files * train_ratio)))
                    val_end = total_files
            else:
                # 3-way split with very limited files -> Distribution skew is inevitable
                print(
                    f"Class '{label}' has only {total_files} file(s). Performing a strict 3-way split "
                    "in a low-file regime will inevitably skew the statistical class distribution across splits. "
                    "Consider setting val_ratio=0.0 for a clean 2-way split.")
                
                if total_files == 4:
                    train_end = 2  # 2 files for Train
                    val_end = 3    # 1 file for Val (Remaining 1 file goes to Test)
                elif total_files == 3:
                    train_end = 1
                    val_end = 2
                else:
                    train_end = 1
                    val_end = total_files

        train_files = set(files_in_label[:train_end])
        val_files = set(files_in_label[train_end:val_end])
        
        # 4. Map Files to Indices
        for idx in class_indices:
            current_file = file_ids[idx]
            if current_file in train_files:
                train_idx.append(idx)
            elif current_file in val_files:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

    # 5. Numpy Arrays
    train_idx_arr = np.array(train_idx, dtype=int)
    val_idx_arr = np.array(val_idx, dtype=int)
    test_idx_arr = np.array(test_idx, dtype=int)
    
    # 6. Final Shuffle
    np.random.shuffle(train_idx_arr)
    np.random.shuffle(val_idx_arr)
    np.random.shuffle(test_idx_arr)
    
    return train_idx_arr, val_idx_arr, test_idx_arr
