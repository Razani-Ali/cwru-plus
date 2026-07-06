Markdown
<div align="center">

# 🚀 CWRU-Plus: The Ultimate Bearing Dataset Pipeline

**An enterprise-grade, ultra-fast, and highly customizable Python library for the Case Western Reserve University (CWRU) Bearing Dataset.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZLASr6GLbxxAsH-HcIwX3KAJgr5BdZb3?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/cwru-plus.svg)](https://badge.fury.io/py/cwru-plus)

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

You can install **CWRU-Plus** directly from PyPI via `pip` (which automatically manages all dependencies like `numpy`, `scipy`, `requests`, and `tqdm`):

```bash
pip install cwru-plus
```
🛠️ For Developers & Contributors
If you want to modify the source code, clone the repository and install it in editable mode:

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

## ☁️ Google Colab Integration & Real-World Benchmarks

Training on the cloud? CWRU-Plus is fully optimized for Google Colab environments. We exploit lightning-fast local ephemeral storage (`/content/`) for heavy processing, while spawning non-blocking background threads (`safe_copy`) to silently sync your massive `.npz` and `.zip` archives to your Google Drive (`/content/drive/MyDrive/`).

### ⏱️ Live Colab Benchmarks (Proven Speed)

Here is the exact execution breakdown measured via `%%time` in a standard Google Colab environment:

| Pipeline Step | Processed Data | CPU Time | Wall Time (Real Waiting) | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **1. Download & Zip** | ~300 MB Raw `.mat` Files | 20.4 s | **1 min 22 s** | Parallel fetching. Only ~16s spent on compression! |
| **2. Ingest & Save** | All Ingested Signals | 1.7 s | **1.55 s** | Lightning-fast unification into a structured `.npz`. |
| **3. Parallel Filtering** | 4th-Order Butter + Median | 55.5 s | **46.7 s** | Over-utilized multi-core filtering (36s raw computation). |
| **4. In-Memory Ingestion** | Full Sliding Windows + Masking | 660 ms | **660 ms** | Zero-copy vectorization. ML-ready in less than a second! |

> **Why the CPU Time vs. Wall Time discrepancy?** In Step 3, CPU time is **55.5s** while Wall time is only **46.7s**. This proves our multi-threaded execution is successfully utilizing multiple cores simultaneously, saving you precious waiting time.

👉 **[Run the interactive, benchmarked End-to-End Pipeline in Colab right now!](https://colab.research.google.com/drive/1ZLASr6GLbxxAsH-HcIwX3KAJgr5BdZb3?usp=sharing)**

---

## 📊 Output Tensor Formats

The `cwru.load()` method is highly flexible. Depending on your configuration, it guarantees strict alignment between signals and metadata:

* **Standard Flattened:** `X.shape = (Total_Windows, Channels, Window_Size)`
* **Time-Partitioned (Cross-Validation Ready):** By setting `num_parts=10`, `X` becomes `(10, Windows_per_Part, Channels, Window_Size)`. Perfect for preventing Data Leakage in time-series!
* **Dynamic Labels:** Choose from `'only types'`, `'types & severity'`, or full structural strings `'types, severity & locations'` like `'OR014@6'`.

---

## 💡 The "Why": Solving the CWRU Bottleneck

As researchers in Condition Monitoring and Fault Diagnosis, we realized that 50-70% of our time was wasted on "Data Engineering" rather than actual Deep Learning research. We built **CWRU-Plus** to eliminate the three biggest bottlenecks in the field:

### 1. The Preprocessing Time-Sink ⏳
*   **The Problem:** Traditional scripts use nested `for` loops and naive file reading to parse hundreds of `.mat` files, which can take hours.
*   **The CWRU-Plus Solution:** We implemented high-performance Multi-threading for file ingestion and NumPy's zero-copy `sliding_window_view` for instantaneous window extraction. What used to take hours now takes less than 3 seconds.

### 2. Data Leakage & Alignment Risks ⚠️
*   **The Problem:** Manually aligning metadata (HP, Fault Type, Severity) with fragmented time-series signals is highly error-prone, often leading to silent data leakage and ruined experiments.
*   **The CWRU-Plus Solution:** Our engine uses an "Atomic Record" strategy. Signals and metadata are locked together into a single dictionary during parallel extraction, guaranteeing 100% strict row-level alignment across the entire dataset. 

### 3. Rigid DSP Pipelines 🧱
*   **The Problem:** Existing repositories hardcode their preprocessing steps, making it a nightmare to apply custom noise removal or filtering.
*   **The CWRU-Plus Solution:** Through **Inversion of Control (IoC)**, CWRU-Plus completely decouples the data loading from the signal processing. You can inject isolated, custom linear or non-linear DSP filters per individual channel (e.g., Drive End vs. Fan End) without touching the core library.

---

## 🔮 What's Next? We want your feedback!

We built CWRU-Plus to accelerate our own research, but this library is for the community. We want to know what would make your research even faster! 

**What features would you like to see next?**
*   *Native PyTorch `Dataset`/`DataLoader` wrappers?*
*   *Built-in Time-Frequency transformations (STFT, Wavelets, Hilbert-Huang)?*
*   *Support for other bearing datasets (PU, IMS, XJTU-SY)?*

👉 **[Open an Issue](https://github.com/Razani-Ali/cwru-plus/issues)** on GitHub and tell us what you want us to build next!

🟢🟡🔴We Are working on support for **MAFAULDA Bearing Dataset**, it would be released soon!

---

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are highly welcome! Feel free to open an issue or submit a Pull Request.

This project is open-source and licensed under the **MIT License**.
