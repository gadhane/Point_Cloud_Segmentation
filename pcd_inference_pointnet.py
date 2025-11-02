"""
Code for Inference & Evaluation for PointNet Point-Cloud Segmentation

- Uses the trained module saved as best_model.pth
- Fast single-tile prediction + interactive visualization (Open3D)
- Qualitative preview and export to TXT/PLY
- Metrics aggregation over a folder (OA, Macro-F1, mIoU, CM)

Run:
    python pcd_inference_pointnet.py
"""

from __future__ import annotations
import os
from pathlib import Path
from glob import glob
import time
import mock
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch

# Import your model + utilities from the training script
from pcd_segmentation_pointnet import PointNet, PointCloudClassifier, cloud_loader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# ========= Configuration (edit to match training) =========
PROJECT_DIR = Path("data")
TEST_GLOB   = str(PROJECT_DIR / "test" / "*.txt")
CHECKPOINT  = "best_model.pth"

CLOUD_FEATURES = "xyzrgbi"  # features used by the loader/model
CLASS_NAMES    = ['unclassified', 'vegetation', 'ground', 'buildings']
GPU_USE        = 0   # set to 1 only if your PointCloudClassifier.run is device-aware

# Args 
def build_args() -> mock.Mock:
    args = mock.Mock()
    args.n_classes      = len(CLASS_NAMES)
    args.input_feats    = CLOUD_FEATURES
    args.subsample_size = 2048
    args.cuda           = GPU_USE
    return args


# ========= Model loading =========
def load_model(checkpoint_path: str | Path, args: mock.Mock, device: torch.device) -> PointNet:
    """
    Build PointNet with the same architecture
    """
    model = PointNet(
        MLP_1=[32, 32],
        MLP_2=[32, 64, 256],
        MLP_3=[128, 64, 32],
        n_class=args.n_classes,
        input_feat=len(args.input_feats),          # 7 for "xyzrgbi"
        subsample_size=args.subsample_size,
        cuda=0,                                    
    )
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ========= Prediction (single tile) =========
def predict_tile(
    tile_path: str | Path,
    model: PointNet,
    pcc: PointCloudClassifier,
    features_used: str = "xyzrgbi",
    device: torch.device | None = None,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Predict per-point labels for a single tile and return an Open3D point cloud with points
    (colors not set here) and the predicted integer labels.

    Returns:
        pcd  : o3d.geometry.PointCloud (points only; colors can be assigned later)
        labels: np.ndarray of shape [N,] with predicted class ids
    """
    # Load features & ground-truth
    cloud, gt = cloud_loader(str(tile_path), features_used)

    
    # By default we keep CPU to avoid device mismatch with PCC internals.
    if device is not None and device.type == "cuda":
        cloud = cloud.to(device)

    # Run prediction
    t0 = time.time()
    with torch.no_grad():
        logits = pcc.run(model, [cloud])          # shape [N, C]
        labels = logits.argmax(dim=1).squeeze().cpu().numpy()
    print(f"[{Path(tile_path).name}] Prediction time: {time.time() - t0:.2f}s")

    # Build Open3D point cloud
    xyz = np.asarray(cloud[0:3].cpu() if torch.is_tensor(cloud) else cloud[0:3]).T  # [N,3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd, labels


# ========= Visualization helpers =========
def colorize_pointcloud_with_labels(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    cmap_name: str = "tab20",
) -> o3d.geometry.PointCloud:
    """
    Assign colors to a point cloud based on integer labels using a Matplotlib colormap.
    """
    if labels.size == 0:
        return pcd

    max_label = labels.max()
    denom = max_label if max_label > 0 else 1
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(labels / denom)[:, :3]          # RGB in [0,1]
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def show_pointcloud(pcd: o3d.geometry.PointCloud) -> None:
    """
    Quick interactive viewer (ESC to close).
    """
    try:
        pcd.estimate_normals(fast_normal_computation=True)
    except:
        pass
    o3d.visualization.draw_geometries([pcd])


# ========= Export helpers =========
def export_predictions_txt_ply(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    out_txt: str | Path,
    out_ply: str | Path,
) -> None:
    """
    Save predicted point cloud:
      - TXT with X Y Z label (space-separated)
      - PLY colored by label, for easy viewing in MeshLab/CloudCompare
    """
    out_txt = Path(out_txt)
    out_ply = Path(out_ply)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(pcd.points)                   # [N,3]
    stacked = np.hstack([xyz, labels.reshape(-1, 1)])
    np.savetxt(str(out_txt), stacked, fmt="%.4f", delimiter=" ")

    # If PLY colors are not set yet, set some default colors
    if not pcd.has_colors():
        colorize_pointcloud_with_labels(pcd, labels)

    o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=True)


# =========qualitative loop =========
def preview_first_k(
    inference_files: list[str],
    k: int,
    model: PointNet,
    pcc: PointCloudClassifier,
    features_used: str,
) -> None:
    """
    Show first k tiles interactively (one window per tile).
    """
    for tile_path in inference_files[:k]:
        pcd, labels = predict_tile(tile_path, model, pcc, features_used=features_used)
        colorize_pointcloud_with_labels(pcd, labels)
        show_pointcloud(pcd)

def evaluate_folder_with_gt(inference_files, model, pcc, features_used, class_names,
                            exclude_label_name="unclassified"):
    excl_idx = class_names.index(exclude_label_name) if exclude_label_name in class_names else None
    kept_label_indices = [i for i in range(len(class_names)) if i != excl_idx] if excl_idx is not None else list(range(len(class_names)))
    kept_label_names   = [class_names[i] for i in kept_label_indices]

    all_pred, all_gt = [], []

    for tile_path in inference_files:
        cloud, gt = cloud_loader(tile_path, features_used)
        with torch.no_grad():
            logits = pcc.run(model, [cloud])
            pred = logits.argmax(dim=1).cpu().numpy()

        gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else gt

        # drop excluded GT points
        if excl_idx is not None:
            mask = (gt_np != excl_idx)
            gt_np = gt_np[mask]
            pred  = pred[mask]

        all_pred.append(pred)
        all_gt.append(gt_np)

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_gt, axis=0)

    if y_true.size == 0:
        return {
            "overall_accuracy_%": 0.0,
            "macro_F1_%": 0.0,
            "mIoU_%": 0.0,
            "per_class_IoU_%": {n: 0.0 for n in kept_label_names},
            "confusion_matrix": np.zeros((len(kept_label_indices), len(kept_label_indices)), dtype=int),
            "classification_report_text": "No valid samples after excluding the specified label."
        }

    cm = confusion_matrix(y_true, y_pred, labels=kept_label_indices)

    eps = 1e-6
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    iou = TP / (TP + FP + FN + eps)
    miou = float(np.nanmean(iou) * 100.0)

    metrics = {
        "overall_accuracy_%": accuracy_score(y_true, y_pred) * 100.0,
        "macro_F1_%": f1_score(y_true, y_pred, labels=kept_label_indices, average="macro", zero_division=0) * 100.0,
        "mIoU_%": miou,
        "per_class_IoU_%": dict(zip(kept_label_names, (iou * 100.0).tolist())),
        "confusion_matrix": cm,
        "classification_report_text": classification_report(
            y_true, y_pred, labels=kept_label_indices, target_names=kept_label_names, digits=3, zero_division=0
        )
    }
    return metrics

def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_png: str | Path) -> str:
    """
    Save a normalized confusion matrix image to disk.
    """
    out_png = str(out_png)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(cm_sum, 1, None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ========= Main =========
def main():
    # 1) Collect files
    inference_list = sorted(glob(TEST_GLOB))
    if not inference_list:
        raise FileNotFoundError(f"No TXT files found for pattern: {TEST_GLOB}")

    # 2) Build args & device
    args = build_args()
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        print("Ensure PointCloudClassifier.run is device-aware. Otherwise set GPU_USE=0.")
    print(f"Using device: {device.type.upper()}")

    # 3) Init PCC and model
    pcc   = PointCloudClassifier(args)   
    model = load_model(CHECKPOINT, args, device)

    # 4) Single-tile example (interactive visualization)
    example_tile = inference_list[min(8, len(inference_list)-1)]
    pcd, labels = predict_tile(example_tile, model, pcc, features_used=args.input_feats, device=None)
    colorize_pointcloud_with_labels(pcd, labels)
    show_pointcloud(pcd)

    # 5) Quick qualitative pass on first 5 tiles
    for tile_path in inference_list[:5]:
        pcd, labels = predict_tile(tile_path, model, pcc, features_used=args.input_feats, device=None)
        colorize_pointcloud_with_labels(pcd, labels)
        show_pointcloud(pcd)

    # 6) Export predictions for one tile (TXT + PLY)
    export_dir = Path("preds")
    export_dir.mkdir(exist_ok=True)
    export_txt = export_dir / f"{Path(example_tile).stem}_pred.txt"
    export_ply = export_dir / f"{Path(example_tile).stem}_pred.ply"
    export_predictions_txt_ply(pcd, labels, export_txt, export_ply)
    print(f"Saved: {export_txt} and {export_ply}")

    # 7) Full metrics over the test folder 
    metrics = evaluate_folder_with_gt(inference_list, model, pcc, args.input_feats, CLASS_NAMES)
    print("\n===== Test Metrics =====")
    print(f"Overall Accuracy: {metrics['overall_accuracy_%']:.2f}%")
    print(f"Macro F1:         {metrics['macro_F1_%']:.2f}%")
    print(f"mIoU:             {metrics['mIoU_%']:.2f}%")
    print("\nPer-class IoU (%):")
    for name, iou in metrics['per_class_IoU_%'].items():
        print(f"  {name:>15}: {iou:6.2f}")
    print("\nClassification report:")
    print(metrics["classification_report_text"])
    cm_png = save_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES[1:], "confusion_matrix.png")
    print(f"Saved confusion matrix to: {cm_png}")


if __name__ == "__main__":
    main()

