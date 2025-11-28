# Clustering & Anomaly Detection Assignments

This repository contains a comprehensive suite of clustering and anomaly detection implementations, ranging from classical algorithms built from scratch to state-of-the-art multimodal Large Language Model (LLM) embeddings.

The projects are designed to be run in **Google Colab**.

## 📂 Project Structure

| Part | Assignment | Description | Key Libraries |
| :--- | :--- | :--- | :--- |
| **A** | **K-Means from Scratch** | Implementation of K-Means using pure `numpy` without sklearn's core algorithm. Includes visualization of centroids and clusters. | `numpy`, `matplotlib` |
| **B** | **Hierarchical Clustering** | Agglomerative clustering using dendrograms to visualize potential cluster counts on synthetic data. | `scipy`, `sklearn` |
| **C** | **Gaussian Mixture Models** | GMM implementation demonstrating soft-clustering and handling of elliptical data shapes. | `sklearn` |
| **D** | **DBSCAN Clustering** | Density-based clustering using **PyCaret** to identify arbitrary shapes (e.g., Moons dataset) and outliers. | `pycaret` |
| **E** | **Anomaly Detection** | Multivariate anomaly detection using **PyOD** (KNN) with heatmap visualizations of anomaly scores. | `pyod` |
| **F** | **Time Series Clustering** | Clustering stock market trends and the UCR "Trace" dataset using **Dynamic Time Warping (DTW)**. | `tslearn`, `yfinance` |
| **G** | **Document Clustering** | Semantic clustering of the "20 Newsgroups" dataset using SOTA **BERT embeddings** (`all-MiniLM-L6-v2`). | `sentence-transformers` |
| **H** | **Image Clustering** | Multimodal clustering using Meta's **ImageBind** to group images (Dogs, Cars, Fruits) by semantic meaning. | `ImageBind`, `torch` |
| **I** | **Audio Clustering** | Audio embedding extraction and clustering using **ImageBind** on environmental sounds (Animals, Transport, Instruments). | `ImageBind`, `torchaudio` |

---

## 🚀 How to Run

These notebooks are optimized for **Google Colab**. 

### 1. General Setup
Most notebooks (Parts A-G) run on the standard CPU runtime.
* Open the notebook in Colab.
* Run the first cell to install dependencies (e.g., `!pip install pycaret`, `!pip install pyod`).

### 2. GPU Requirement (Parts H & I)
For **ImageBind** assignments (Part H and Part I), you **must** enable the GPU runtime.
1.  Go to **Runtime** > **Change runtime type**.
2.  Select **T4 GPU**.
3.  Run the notebook cells in order.

---

## 🛠️ Dependencies & Installation

If running locally, you will need the following major libraries. Note that **PyCaret** and **ImageBind** have specific version requirements (Python 3.8-3.11 for PyCaret, and specific `timm` versions for ImageBind).

```bash
# Core
pip install numpy pandas matplotlib seaborn scikit-learn

# Domain Specific
pip install pycaret          # Part D
pip install pyod             # Part E
pip install tslearn yfinance # Part F
pip install sentence-transformers # Part G

# ImageBind (Parts H & I)
# Requires cloning the repository and specific dependencies:
git clone [https://github.com/facebookresearch/ImageBind.git](https://github.com/facebookresearch/ImageBind.git)
pip install timm==0.6.13
pip install soundfile
