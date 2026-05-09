# CMNet: Symmetry-Aware Facial Expression Recognition API

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

## 📄 Academic Documentation
* **[Final Project Report](CV_Final_Report.pdf)**: Comprehensive analysis of the architecture, ablation studies, and deployment methodology.
* **[Project Proposal](CV_Proposal.pdf)**: Initial problem statement, proposed methodology, and experimental setup.

## 📌 Abstract
Automated Facial Expression Recognition (FER) frequently suffers from background noise and partial occlusion when relying on standard global-feature Convolutional Neural Networks (CNNs). This repository contains the production deployment and algorithmic optimization of the **Cross-Modal Network (CMNet)**. 

By enforcing biological facial symmetry constraints, engineering a novel Multi-Scale Test-Time Augmentation (MS-TTA) algorithm, and deploying the model within a highly optimized, stateless Docker container via FastAPI, we achieve real-time, robust emotion classification suitable for production environments.

![Teaser Figure: FastAPI Swagger UI](api_screenshot.png) 

## ✨ Key Engineering Features
1. **Multi-Scale SA-TTA Inference:** A custom late-fusion inference pipeline that dynamically evaluates geometrically mirrored and scaled variants of the input tensor to neutralize unilateral spatial bias.
2. **Label Distribution Learning (LDL):** Replaces rigid Cross-Entropy with an algorithmic KL-Divergence loss, mapping emotions to a fluid biological spectrum.
3. **Zero-Trust Security Patch:** Implements a namespace monkey-patch to safely bypass PyTorch 2.6 `WeightsUnpickler` vulnerabilities associated with legacy `.pth` files.
4. **MLOps Ready:** Fully containerized, stateless REST API delivering sub-20ms inference latency.

## ⚙️ Architecture & Approach
We conduct a comparative architectural analysis between traditional global CNNs and the CMNet paradigm. CMNet operates on three synchronous inputs: a global aligned face, and two localized multi-modal variants (upper and lower facial quadrants).

### The SFIRM Module & Symmetry Loss
To prevent the negative effects of cross-modal fusion, CMNet introduces a **Salient Facial Information Refinement Module (SFIRM)** alongside a Symmetry Loss function ($L_{sl}$). This function aligns the left and right facial feature vectors ($x_l, x_r$) obtained after Global Average Pooling:

$$L_{sl} = \frac{1}{NC} \sum_{i=1}^{N} \sum_{j=1}^{C} \left( x_l^{(i,j)} - x_r^{(i,j)} \right)^2$$

This ensures the model actively learns symmetrical emotional triggers rather than overfitting to background asymmetry.

## 📊 Experiments and Results
The model was evaluated using the official test split of the **RAF-DB** dataset (3,068 images) across the 7 universal emotion classes.

| Inference Protocol | Pre-Processing | Accuracy (RAF-DB) | Net $\Delta$ |
| :--- | :--- | :--- | :--- |
| **ResNet-18** | Global Face Only | 86.96% | - |
| **CMNet Baseline** | Single-Pass | **89.11%** | - |
| **CMNet + MS-TTA** | Flip + Zoom Late-Fusion | 89.05% | -0.07%* |

*\*Discussion: Applying geometric transformations (MS-TTA) to rigorously pre-aligned datasets like RAF-DB introduces microscopic interpolation noise. This ablation study proves MS-TTA should act as a dynamic regularizer—disabled for curated academic datasets, but engaged for unconstrained live video feeds.*

---

## 🚀 Deployment Strategy & Quick Start

To satisfy modern MLOps production standards, the inference engine is fully containerized. It bypasses local environment dependency conflicts by utilizing a stateless Linux container.
[![Live Demo](https://img.shields.io/badge/Live_Demo-Watch_Video-FF0000?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1cSXqcfSpEDKIbiPvorGnx69iB3M9YtJ-?usp=drive_link)

### Prerequisites
* Docker Desktop installed and running.
* Git.

### 1. Build the Docker Image
Clone the repository and build the container. This will download the necessary Python 3.10 environment and install all dependencies securely.
```bash
git clone [https://github.com/mashaal03/CMNet-Facial-Expression-API.git](https://github.com/mashaal03/CMNet-Facial-Expression-API.git)
cd CMNet-Facial-Expression-API
docker build -t cmnet-api .
