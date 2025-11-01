# 3D Point Cloud Semantic Segmentation (PointNet)

A lightweight project for **per-point semantic segmentation** of unstructured 3D point clouds using a **PointNet-style** architecture. It trains on plain `.txt` tiles and produces colored `.ply` point clouds plus full evaluation metrics.

---

## Overview

* **Task:** Semantic segmentation of each point into classes like *ground*, *vegetation*, *buildings*, etc.
* **Approach:** **PointNet** with shared per-point MLPs (`Conv1d` with `kernel_size=1`) + **global max pooling** for context aggregation + per-point classification head.
* **Why PointNet?** It treats a point cloud as an **unordered set**, avoiding heavy voxelization (3D grids) and preserving point precision.
* **Upsampling step:** The model predicts on a random **subsample** and then **propagates logits** to all points using **1-nearest neighbor** (KNN) mapping.

---

## Techniques & Design Choices

* **Per-point MLP via 1×1 Conv1D:** Implements the PointNet idea of applying the same MLP to every point (permutation-invariant).
* **Global feature:** `MaxPool1d` over points to form a shape/scene descriptor concatenated back to per-point features.
* **Subsample → Predict → 1-NN Upsample:**

  * Subsample to `subsample_size` points for speed.
  * Predict per-point logits on the subset.
  * **Upsample with `NearestNeighbors` (scikit-learn)** to assign each original point the logits of its nearest sampled point.
* **Metrics:** accuracy, macro-F1, per-class IoU, mIoU, and a confusion matrix (with the option to **exclude “unclassified”** from averaging/reporting to avoid ill-defined warnings).

---

## Data & Folder Structure

```
project/
├─ pcd_segmentation_pointnet.py        # training (this repo’s main training file)
├─ pcd_inference_pointnet.py           # inference/evaluation/visualization (standalone)
├─ data/
│  ├─ train/
│  │   ├─ *.txt
│  └─ test/
│      ├─ *.txt
└─ preds/                              # generated: exported predictions (TXT/PLY)
```

### TXT tile format (expected by `cloud_loader`)

`np.loadtxt(tile).transpose()` is used; the loader expects rows to be features after transpose:

* **XYZ:** rows `0:3`
* **RGB:** rows `3:6` (optional if you include `"rgb"`)
* **Intensity:** row `-2` (second-to-last) if you include `"i"`
* **Ground truth label:** row `-1` (last) as integer class id

> If your files differ, adjust `cloud_loader` accordingly.

---

## Requirements

* Python 3.9+
* PyTorch
* scikit-learn
* Open3D
* matplotlib
* torchnet
* tqdm
* (optional) CUDA toolkit & drivers for GPU

Install (example):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install scikit-learn open3d matplotlib torchnet tqdm
```

---

## Training

`pcd_segmentation_pointnet.py` contains the training loop and is minimally modularized with a `main()`.

```bash
python pcd_segmentation_pointnet.py
```

Key defaults (edit inside `build_args()` if present, or in the script):

* `data_path="data/"`
* `input_feats="xyzrgbi"` → 7 input channels
* `n_classes=4`  (e.g., `['unclassified', 'vegetation', 'ground', 'buildings']`)
* `subsample_size=2048`
* `mlp1=[32,32]`, `mlp2=[32,64,256]`, `mlp3=[128,64,32]`
* Optimizer: Adam (`lr=1e-3`), `weight_decay=1e-4`
* LR schedule: milestones `[20, 40]`
* Saves the best checkpoint to **`best_model.pth`** (by validation accuracy)

---

##  Inference & Evaluation

The inference script (e.g., `pcd_inference_pointnet.py`) **loads `best_model.pth`** and supports:

* **Per-tile prediction** and **interactive Open3D visualization**
* **Export** of predicted point clouds:

  * `preds/<tile>_pred.txt` → `X Y Z label`
  * `preds/<tile>_pred.ply` → colored point cloud for CloudCompare/MeshLab
* **Metrics** computed across the test set:

  * Overall Accuracy, Macro-F1, per-class IoU, mIoU
  * Confusion Matrix (PNG)
  * Option to **exclude “unclassified”** from all metrics to avoid warnings

Run:

```bash
python pcd_inference_pointnet.py
```
---

##  Sample Outputs

> You can drop your visuals here.

* **Colored prediction (PLY screenshots)**

  * `images/pred_scene_01.png`
  * `images/pred_scene_02.png`

* **Confusion Matrix**

  * `images/confusion_matrix.png`

---

## Citation / Acknowledgements

* Inspired by **PointNet**: *Qi et al., “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,” CVPR 2017.*
* I want to thank [Prof. Florent Poux](https://learngeodata.eu/) for his amazing 3D computer vision courses and tutorials. His course, tutorials and data helped to learn 3D vision from scratch including how to implement PointNet and use for Point cloud classificaiton. 

---
