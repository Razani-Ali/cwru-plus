Markdown
<div align="center">

# 🚀 CWRU-Plus: The Ultimate Bearing Dataset Pipeline

**An enterprise-grade, ultra-fast, and highly customizable Python library for the Case Western Reserve University (CWRU) Bearing Dataset.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZLASr6GLbxxAsH-HcIwX3KAJgr5BdZb3?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🌟 Why CWRU-Plus?
Stop wasting days writing loops to parse messy `.mat` files! **CWRU-Plus** is built for modern Machine Learning and Deep Learning researchers who need clean, ML-ready tensors *instantly*. 

Whether you are looking for pure simplicity, or you are conducting complex **Domain Adaptation** research requiring channel-specific signal processing, CWRU-Plus handles it with unprecedented speed.

### 🔥 Key Superpowers

*   **🪄 Magic in 3 Lines:** Go from raw files to fully aligned PyTorch/TensorFlow-ready 3D tensors `(Samples, Channels, Sequence_Length)` in just three lines of code.
*   **⚡ Blazing Fast Extraction:** Thanks to zero-copy memory mapping (`sliding_window_view`) and NumPy vectorization, compiling overlapping windows for the entire dataset takes **less than 3 seconds**.
*   **🎛️ Parallel Custom Filtering:** Need to apply a 4th-order Butterworth filter to the Drive End (DE) and a median filter to the Fan End (FE)? Our parallel engine processes the entire dataset in **under 1 minute**.
*   **☁️ Cloud Native (Colab & Drive):** Built-in daemon threads automatically back up your massive datasets to Google Drive in the background without freezing your Colab runtime.
*   **🔓 100% Free & Open-Source:** Fully customizable under the MIT License.

---

## 🚀 Quick Start: The "3-Line" Promise

Forget complex data engineering. Once installed, getting your ML tensors is this simple:

```python
import cwru

# 1. Download & Ingest automatically
cwru.download(CWRUfs=12, download_path="data/raw")
cwru.ingest(base_path="data/raw", output_file_path="data/dataset")

# 2. Extract ML-ready tensors in < 3 seconds!
(X, Y), metadata = cwru.load(npz_path="data/dataset.npz", window_size=2048, step_size=512)

print(f"Data Shape: {X.shape} | Labels Shape: {Y.shape}")
```

---

## ⚙️ Installation

Clone the repository and install it in editable mode (which automatically installs dependencies like `numpy`, `scipy`, `requests`, and `tqdm`):

```bash
git clone https://github.com/Razani-Ali/cwru-plus.git
cd cwru-plus
pip install -e .

```

---

## 🎛️ Advanced: Channel-Specific Signal Filtering

CWRU-Plus offers **Inversion of Control (IoC)**, allowing you to inject completely different linear or non-linear DSP filters into different sensor channels. 

Using our multi-threaded backend, this heavy computation finishes in **seconds**:  

```python
import scipy.signal as sig
import cwru

# Define your custom filters
def de_filter(signal):
    b, a = sig.butter(4, 0.1, btype='low')
    return sig.filtfilt(b, a, signal) # Zero-phase filtering

def fe_filter(signal):
    return sig.medfilt(signal, kernel_size=5) # Non-linear impulsive noise removal

# Map filters to specific channels
my_filters = {
    "DE": de_filter,
    "FE": fe_filter,
    "BA": None # Keep Base Accelerometer raw
}

# Apply across the entire dataset in parallel
cwru.filter(filters=my_filters, input_npz="data/dataset.npz", output_npz="data/filtered.npz")

```

---

## 🔬 Advanced: Domain Adaptation & Cross-Domain Splits

If you are conducting **Domain Generalization** or **Domain Adaptation** research (e.g., training on specific loads and testing on others), CWRU-Plus makes it incredibly easy.

The `cwru.load()` function returns perfectly aligned metadata vectors (`Severities`, `HorsePowers`, `Locations`). You can use standard NumPy masking to instantly split your dataset into distinct domains:

```python
import cwru

# Load the entire dataset
(X, Y), (Severities, HorsePowers, Locations, sensors) = cwru.load("data/dataset.npz")

# Define your source and target domains based on HorsePower (HP) loads
source_mask = (HorsePowers <= 2) # Train on 0, 1, and 2 HP
target_mask = (HorsePowers == 3) # Test strictly on 3 HP

# Split the tensors using NumPy masking
X_train, Y_train = X[source_mask], Y[source_mask]
X_test, Y_test = X[target_mask], Y[target_mask]

print(f"Source Domain (0-2 HP) Shape: {X_train.shape}")
print(f"Target Domain (3 HP) Shape: {X_test.shape}")

```

---

## ☁️ Google Colab Integration

Training on the cloud? CWRU-Plus is optimized for Google Colab environments. We use lightning-fast local Colab storage (`/content/`) for processing, while spawning safe background threads (`safe_copy`) to silently sync your massive `.npz` and `.zip` archives to your Google Drive (`/content/drive/MyDrive/`).

👉 **[Run the interactive End-to-End Pipeline in Colab right now!](https://colab.research.google.com/drive/1ZLASr6GLbxxAsH-HcIwX3KAJgr5BdZb3?usp=sharing)**

---

## 📊 Output Tensor Formats

The `cwru.load()` method is highly flexible. Depending on your configuration, it guarantees strict alignment between signals and metadata:

* **Standard Flattened:** `X.shape = (Total_Windows, Channels, Window_Size)`
* **Time-Partitioned (Cross-Validation Ready):** By setting `num_parts=10`, `X` becomes `(10, Windows_per_Part, Channels, Window_Size)`. Perfect for preventing Data Leakage in time-series!
* **Dynamic Labels:** Choose from `'only types'`, `'types & severity'`, or full structural strings `'types, severity & locations'` like `'OR014@6'`.

---

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are highly welcome! Feel free to open an issue or submit a Pull Request.

This project is open-source and licensed under the **MIT License**.
