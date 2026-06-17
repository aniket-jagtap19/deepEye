# deepEye - Anomaly Detection for Quality Control

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9.13%2B-green)]()
[![PyTorch](https://img.shields.io/badge/pytorch-1.13.1%2B-orange)]()
[![Torchvision](https://img.shields.io/badge/torchvision-0.14.1%2B-red)]()

</div>

---

## Introduction

This project implements a simple **Autoencoder-based Anomaly Detection** system for industrial quality control. The model is trained only on **defect-free images** from the **MVTec Anomaly Detection Dataset** and learns the normal appearance of objects or textures.

During testing, the model detects defects by comparing the original image with its reconstructed version. The **Structural Similarity Index Measure (SSIM)** is used both as a loss function and as an evaluation metric.

SSIM implementation used in this project:

* https://github.com/VainF/pytorch-msssim

---

## Dataset

The project uses the **MVTec Anomaly Detection Dataset**, which contains:

* 12 different object and texture categories
* 500+ images in total
* Various types of industrial defects

You can download the dataset from:

https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Directory Structure

Place the downloaded dataset inside the `data/` folder as shown below:

```text
data/
├── metal_nut
│   ├── ground_truth
│   │   ├── bent
│   │   ├── color
│   │   ├── flip
│   │   └── scratch
│   │
│   ├── test
│   │   ├── bent
│   │   ├── color
│   │   ├── flip
│   │   ├── scratch
│   │   └── good
│   │
│   └── train
│       └── good
│
└── ...
```

---

## Dependencies

Main libraries used:

```txt
torch >= 2.0.0
torchvision == 0.14.1
numpy == 1.22
```

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## How It Works

1. Train the Autoencoder using only normal (good) images.
2. The model learns the normal structure and appearance of the objects.
3. During testing, the image is reconstructed by the Autoencoder.
4. SSIM is used to compare the original and reconstructed images.
5. Regions with low similarity are identified as anomalies or defects.

---
