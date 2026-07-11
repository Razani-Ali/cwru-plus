import random
import numpy as np
from typing import Tuple, Dict, Optional

class FewShotSampler:
    """
    A high-performance, strategic sampler designed for Few-Shot and Meta-Learning tasks.
    Adapted for pre-windowed 3D tensors.
    
    It decouples the ML algorithm from underlying string labels by mapping requested 
    numerical class indices (e.g., 0, 1, 2) directly to their database string names,
    returning clean, single-integer target labels ideal for CrossEntropyLoss.
    """
    def __init__(self, X_base: np.ndarray, Y_base: np.ndarray, 
                 numeric_to_string: Dict[int, str],
                 meta_base: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (None, None, None)):
        """
        Initializes the Few-Shot Sampler with pre-windowed arrays.

        Args:
            X_base (np.ndarray): Foundational 3D tensor of shape [Total_Windows, Channels, Length].
            Y_base (np.ndarray): Master 1D array of original string labels.
            numeric_to_string (Dict[int, str]): Map dictionary translating integers to strings.
                                                e.g., {0: 'normal', 1: 'imbalance', 2: 'misalignment'}
            meta_base (Tuple, optional): Bound master metadata (Severity, HP, Location).
                                         Defaults to (None, None, None).
        """
        self.X = X_base
        self.Y = Y_base
        self.numeric_to_string = numeric_to_string
        
        # Unpack the 3 metadata arrays
        self.Severity, self.HP, self.Location = meta_base
        
        # Extract physical mappings and class capacities directly from pre-windowed rows
        self.indices_by_class = self._map_instances_to_classes()
        self.label_frequencies = self._calculate_label_frequencies()

    def _map_instances_to_classes(self) -> dict:
        """Responsibility: Maps strict row indices based on string categories."""
        mapping = {}
        for i, label in enumerate(self.Y):
            if label not in mapping:
                mapping[label] = []
            mapping[label].append(i)
        return mapping

    def _calculate_label_frequencies(self) -> dict:
        """Responsibility: Calculates exact total windows available per category."""
        return {
            label: len(indices) for label, indices in self.indices_by_class.items()
        }

    def reset_seed(self, seed: int):
        """Responsibility: Controls the pseudo-random seed generator state."""
        random.seed(seed)

    def _validate_inputs(self, target_numeric_classes: Tuple[int, ...], samples_per_class: Tuple[int, ...]):
        """Responsibility: Structural validation of requested numerical categories and database bounds."""
        if len(target_numeric_classes) != len(samples_per_class):
            raise ValueError("❌ Shape Error: The length of target numeric classes must match samples per class!")

        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            if num_label not in self.numeric_to_string:
                raise ValueError(f"❌ Key Error: Numerical class ID '{num_label}' is missing from the injected map!")
            
            string_name = self.numeric_to_string[num_label]
            if string_name not in self.indices_by_class:
                raise ValueError(f"❌ Database Error: String class '{string_name}' was not found during ingestion!")
                
            if self.label_frequencies[string_name] < required_samples:
                raise ValueError(
                    f"❌ Capacity Error: Class '{string_name}' has {self.label_frequencies[string_name]} windows, "
                    f"which is less than the requested {required_samples} samples!"
                )

    def _sample_single_class(self, num_label: int, required_samples: int) -> Tuple[list, list, list, list, list]:
        """Responsibility: Random selection of pre-windowed instances and data stream assembly for one class ID."""
        sampled_x, sampled_y = [], []
        sampled_sev, sampled_hp, sampled_loc = [], [], []
        
        string_name = self.numeric_to_string[num_label]
        available_indices = self.indices_by_class[string_name]
        
        # Directly sample the exact row indices since the data is already flattened into 3D
        chosen_indices = random.sample(available_indices, required_samples)

        for idx in chosen_indices:
            sampled_x.append(self.X[idx])
            sampled_y.append(num_label) # Store the integer class ID directly
            
            # Safely capture metadata if it exists
            if self.Severity is not None:
                sampled_sev.append(self.Severity[idx])
            if self.HP is not None:
                sampled_hp.append(self.HP[idx])
            if self.Location is not None:
                sampled_loc.append(self.Location[idx])

        return sampled_x, sampled_y, sampled_sev, sampled_hp, sampled_loc

    def _post_process_and_shuffle(self, s_x: list, s_y: list, s_sev: list, s_hp: list, s_loc: list) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Responsibility: Unified integration, random shuffling, and formatting to production arrays."""
        combined = list(zip(s_x, s_y, s_sev, s_hp, s_loc))
        random.shuffle(combined)

        X_final = np.array([item[0] for item in combined])
        Y_final = np.array([item[1] for item in combined], dtype=np.int64) # 1D array of integer labels
        
        # Pack the updated 3-element metadata
        Sev_final = np.array([item[2] for item in combined], dtype=object) if self.Severity is not None else None
        HP_final = np.array([item[3] for item in combined], dtype=object) if self.HP is not None else None
        Loc_final = np.array([item[4] for item in combined], dtype=object) if self.Location is not None else None
        
        return X_final, Y_final, (Sev_final, HP_final, Loc_final)

    def sample(self, target_numeric_classes: Tuple[int, ...], 
               samples_per_class: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Orchestrates the entire execution pipeline. Accepts numeric IDs and drops integer array labels.
        
        Args:
            target_numeric_classes: Tuple of integer IDs representing the chosen classes (e.g., (0, 2))
            samples_per_class: Parallel tuple specifying required window slices per ID (e.g., (50, 5))
            
        Returns:
            X_final: Matched 3D task array [Samples, Channels, WindowSize]
            Y_final: 1D NumPy int64 array containing explicit class index integers [Samples,]
            Meta_final: Packed metadata trackers matching the extracted instances (Severity, HP, Location)
        """
        self._validate_inputs(target_numeric_classes, samples_per_class)

        all_sampled_x, all_sampled_y = [], []
        all_sampled_sev, all_sampled_hp, all_sampled_loc = [], [], []

        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            x_c, y_c, sev_c, hp_c, loc_c = self._sample_single_class(num_label, required_samples)
            all_sampled_x.extend(x_c)
            all_sampled_y.extend(y_c)
            all_sampled_sev.extend(sev_c)
            all_sampled_hp.extend(hp_c)
            all_sampled_loc.extend(loc_c)

        results = self._post_process_and_shuffle(
            all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_hp, all_sampled_loc
        )
        
        return results