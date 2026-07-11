Markdown
<div align="center">

# 🚀 CWRU-Plus: The Ultimate Bearing Dataset Pipeline

**An enterprise-grade, ultra-fast, and highly customizable Python library for the Case Western Reserve University (CWRU) Bearing Dataset.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W4ELtyXa5lfKgNyi6uChexV2oWIlnKI9?usp=sharing)
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

# ==============================================================================
# ⚠️ STEP 1: ONE-TIME DATA SETUP (Download & Ingest)
# Run these two lines ONLY ONCE to build your persistent local database.
# After 'dataset.npz' is generated, you can comment out or remove Step 1 entirely.
# ==============================================================================
cwru.download(CWRUfs=12, download_path="data/raw")
cwru.ingest(base_path="data/raw", output_file_path="data/dataset")

# ==============================================================================
# 🚀 STEP 2: FAST PRODUCTION LOADING (Run repeatedly from here)
# Once you have the permanent .npz file, you ONLY need to execute from this line onwards.
# Ingested arrays are loaded instantly into your pipeline in under 3 seconds!
# ==============================================================================
(X, Y), metadata = cwru.load(npz_path="data/dataset.npz", window_size=2048, step_size=512)

print(f"Data Shape: {X.shape} | Labels Shape: {Y.shape}")
```
---
### 📂 Offline / Local Directory Import (`local_download`)
---
If you have already downloaded the `.mat` files manually (e.g., `105.mat`, `112.mat`) and they are dumped into a single folder, you don't need to re-download them. `CWRU-Plus` can scan your local directory, filter out the specific files based on the requested sampling frequency, and safely copy them to a target directory with our standardized, ML-ready naming convention.

Your original files are never altered or deleted during this process.

```python
from cwru import offline_download

# Safely extract and rename ONLY the 48kHz files from your messy downloads folder
success = offline_download(
    source_path="data/cwru_raw",
    target_path="data/CWRU_Standard",
    CWRUfs=48,             # Explicitly target 48kHz dataset links
    replace_files=False    # Do not overwrite if standard file already exists
)
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
## 🎯 Built-in Few-Shot & Meta-Learning Sampler

Modern fault diagnosis research heavily relies on episodic training configurations like MAML or Prototypical Networks. To bridge the gap between raw data and deep learning architectures, `CWRU-Plus` ships with a high-performance `FewShotSampler`.

Instead of writing manual indexing loops, you can construct fully synchronized episodic tasks instantly. The sampler automatically manages operational metadata (Severity, HP, Location) and returns clean, integer-mapped arrays explicitly optimized for loss functions like PyTorch's `CrossEntropyLoss`.

```python
from cwru.sampler import build_few_shot_sampler

# Initialize the sampler over your pre-windowed dataset
sampler = build_few_shot_sampler(
    X_base=X_data, Y_base=Y_labels,
    numeric_to_string={0: 'Normal', 1: 'Inner_Fault'},
    meta_base=(Severity, HP, Location),
    seed=101
)

# Extract a 2-way, 5-shot episodic task in milliseconds
X_task, Y_task, Metadata = sampler.sample(
    target_numeric_classes=(0, 1), 
    samples_per_class=(5, 5)
)
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

👉 **[Run the interactive, benchmarked End-to-End Pipeline in Colab right now!](https://colab.research.google.com/drive/1W4ELtyXa5lfKgNyi6uChexV2oWIlnKI9?usp=sharing)**

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

---
## Why CWRU-Plus? (Comparison with Alternatives)

While legacy tools on PyPI—such as `cwru`, `py-cwru`, `multivariate-cwru`, and `bearing-python`—laid the foundation for using this dataset, they were built for older Python ecosystems and lack the performance optimization required for modern Deep Learning pipelines. 

Below is a technical matrix showing how **CWRU-Plus** redesigns the entire data engineering layer compared to existing alternatives:

| Feature / Capability | Legacy Packages (`cwru`, `py-cwru`, etc.) | ⚡ CWRU-Plus |
| :--- | :--- | :--- |
| **Python 3.10+ Compatibility** | ❌ No (Many crash on modern `collections` or `numpy` types) | **✅ Yes (Native, production-ready)** |
| **Multi-Threaded Downloading** | ❌ No (Single-threaded, slow legacy sequential mirrors) | **✅ Yes (Parallel high-speed downloads)** |
| **Smart Cache Management** | ❌ No (Re-downloads or fails if archives already exist) | **✅ Yes (Detects, validates, and skips existing files)** |
| **Parallel DSP Filtering (IoC)** | ❌ No (Sequential preprocessing bottlenecks) | **✅ Yes (Multi-core parallelized filter injection)** |
| **Atomic Metadata Extraction** | ❌ No (Only returns raw signals; manual mapping needed) | **✅ Yes (Returns synchronized Horse Powers & Fault Severity vectors)** |
| **Strict Data Leakage Protection**| ❌ No (Manual sliding windows often mix train/test frames) | **✅ Yes (Guaranteed 100% atomic boundary separation)** |
| **Built-in Few-Shot Sampler** | ❌ No (Requires manual implementation of episodic tasks) | **✅ Yes (High-performance episodic task sampler)** |
| **Processing Speed** | ⚠️ Slow (Heavy disk I/O and object creation overhead) | **🚀 Blazing Fast (< 3 seconds via zero-copy NumPy views)** |

🔥🔥🔥

---

### Detailed Benchmarks & Technical Edge
---
#### 1. Smart Local Cache Management
Legacy packages usually struggle if you interrupt a download or want to use previously downloaded files, often resulting in corrupted state errors or redundant web requests. `CWRU-Plus` acts as an intelligent file manager: it automatically scans your `./raw` directory, verifies existing files against the official structural grid, and seamlessly proceeds with ingestion without wasting network bandwidth.

#### 2. Synchronized Meta-Tracking (RPM & Severity)
Most alternative packages strip out or ignore the underlying operational context, leaving you with raw signal matrices. For advanced tasks like **Domain Adaptation** or **Regression-based Remaining Useful Life (RUL)** estimation, you need the operational metadata. `CWRU-Plus` keeps `Severity` and `RPM` bound to each window instance, returning clean, production-ready vectors out of the box.

#### 3. Native Meta-Learning Support (Few-Shot Sampler)
If you are training Prototypical Networks, Relation Networks, or MAML architectures, you typically have to write hundreds of lines of boilerplate code to handle N-way K-shot episodic sampling. `CWRU-Plus` introduces a highly optimized `FewShotSampler` that maps string categories to clean, single-integer target labels ideal for PyTorch `CrossEntropyLoss` at microscopic latencies (under 5ms per batch execution).

---

## 🚀 The Next Generation Machinery Engine is Here!

🔥 **UPDATE:** We have officially expanded our industrial machinery fault diagnosis ecosystem! 

While we previously focused heavily on CWRU, we have officially launched **MAFAULDA-Plus**—a powerhouse data engine designed to handle massive, multi-channel vibration signal processing without crashing your environment.

* 🌐 **GitHub Repository:** [Discover MAFAULDA-Plus on GitHub](https://github.com/Razani-Ali/mafaulda-plus)
* 📦 **PyPI Package:** `pip install mafaulda-plus`

If your research is expanding into complex, cross-domain multi-class fault diagnosis with zero-RAM memory constraints, migrate to **MAFAULDA-Plus** today! 🏁

---

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are highly welcome! Feel free to open an issue or submit a Pull Request.

This project is open-source and licensed under the **MIT License**.
