# =========================
# Minimal modularization for TRAINING ONLY
# Keep inference as-is in its own file.
# =========================

import os
import functools
from glob import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnet as tnt
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.neighbors import NearestNeighbors  # kept because your current logic uses it
from tqdm.auto import tqdm
import mock
import random


#%% Defining the cloud loading function
def cloud_loader(tile_name, features_used):
    cloud_data = np.loadtxt(tile_name).transpose()

    min_f = cloud_data.min(axis=1)
    mean_f = cloud_data.mean(axis=1)
    features = []
    if "xyz" in features_used:
        n_coords = cloud_data[0:3]
        n_coords[0] -= mean_f[0]
        n_coords[1] -= mean_f[1]
        n_coords[2] -= min_f[2]
        features.append(n_coords)
    if "rgb" in features_used:
        n_rgb = cloud_data[3:6]
        features.append(n_rgb)
    if "i" in features_used:
        IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)
        n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
        n_intensity -= np.min(n_intensity)
        features.append(n_intensity)

    gt = cloud_data[-1]
    gt = torch.from_numpy(gt).long()
    cloud_data = torch.from_numpy(np.vstack(features))
    return cloud_data, gt

def cloud_collate(batch):
    clouds, labels = list(zip(*batch))
    labels = torch.cat(labels, dim=0)

    return clouds, labels

class PointNet(nn.Module):
    """"
    PointNet network for Semantic Segmentation
    """
    def __init__(self, MLP_1, MLP_2, MLP_3, n_class = 3,
                 input_feat = 3, subsample_size = 512, cuda = 1):
        super(PointNet, self).__init__()
        self.is_cuda = cuda
        self.subsample_size = subsample_size

        m1  = MLP_1[-1]
        m2 = MLP_2[-1]
        modules = []
        for i in range(len(MLP_1)):
            modules.append(
                nn.Conv1d(
                    in_channels = MLP_1[i-1] if i > 0 else input_feat,
                    out_channels = MLP_1[i],
                    kernel_size = 1,
            ))
            modules.append(nn.BatchNorm1d(MLP_1[i]))
            modules.append(nn.ReLU(True))
        self.mlp1 = nn.Sequential(*modules)

        #MlP 2
        modules = []
        for i in range(len(MLP_2)):
            modules.append(
                nn.Conv1d(
                    in_channels = MLP_2[i-1] if i > 0 else m1,
                    out_channels = MLP_2[i],
                    kernel_size = 1,
            ))
            modules.append(nn.BatchNorm1d(MLP_2[i]))
            modules.append(nn.ReLU(True))
        self.mlp2 = nn.Sequential(*modules)
        #MlP 3
        modules = []
        for i in range(len(MLP_3)):
            modules.append(
                nn.Conv1d(
                    in_channels = MLP_3[i-1] if i > 0 else m1 + m2,
                    out_channels = MLP_3[i],
                    kernel_size = 1,
            ))
            modules.append(nn.BatchNorm1d(MLP_3[i]))
            modules.append(nn.ReLU(True))
        modules.append(
            nn.Conv1d(
                in_channels = MLP_3[-1],
                out_channels = n_class,
                kernel_size = 1,
            ))
        self.mlp3 = nn.Sequential(*modules)
        self.maxpool = nn.MaxPool1d(subsample_size)

        if self.is_cuda:
            self.cuda()
    def forward(self, input):
        if self.is_cuda:
            input = input.cuda()
        f1= self.mlp1(input)
        f2= self.mlp2(f1)
        G = self.maxpool(f2)
        Gf1 = torch.cat((G.repeat(1,1,self.subsample_size), f1),1)
        out = self.mlp3(Gf1)
        return out

#%% Define the Semantic Segmentation Class
class PointCloudClassifier:
    def __init__(self, args):
        # self.sample_size = args.sample_size
        self.sample_size = args.subsample_size
        self.n_input_feats = 3
        if 'i' in args.input_feats:
            self.n_input_feats += 1
        if 'rgb' in args.input_feats:
            self.n_input_feats += 3
        self.n_classes = args.n_classes
        self.is_cuda = args.cuda

    def run(self, model, clouds):
        """
        Input:
        Model = the neural network model
        clouds = list of point clouds to classify
        Output: pred
        """
        n_batch = len(clouds)
        prection_batch = torch.zeros((self.n_classes, 0))
        sampled_clouds = torch.Tensor(n_batch, self.n_input_feats, self.sample_size)

        if self.is_cuda:
            prection_batch = prection_batch.cuda()
        for i_batch in range(n_batch):
            cloud = clouds[i_batch][:, :]
            n_pts = cloud.shape[1]
            replace_flag = n_pts < self.sample_size
            selected_points = np.random.choice(n_pts, self.sample_size, replace=replace_flag)
            sampled_cloud = cloud[:, selected_points]
            sampled_clouds[i_batch, :, :] = sampled_cloud
        sampled_predictions = model(sampled_clouds.float())
        for i_batch in range(n_batch):
            cloud = clouds[i_batch][:3, :]
            sampled_cloud = sampled_clouds[i_batch, :3, :]

            knn = NearestNeighbors(n_neighbors=1,
                                   algorithm='kd_tree').fit(sampled_cloud.cpu().permute(1, 0).numpy())

            dump, closest_point = knn.kneighbors(cloud.permute(1, 0).cpu().numpy())
            closest_point = closest_point.squeeze()

            prediction_full_cloud = sampled_predictions[i_batch, :, closest_point]
            prection_batch = torch.cat((prection_batch, prediction_full_cloud), 1)
        return prection_batch.permute(1, 0)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_train_valid(train_files, valid_ratio=0.2):
    n = len(train_files)
    idx = np.random.choice(n, size=int(valid_ratio * n), replace=False)
    valid = [train_files[i] for i in idx]
    train = [train_files[i] for i in range(n) if i not in idx]
    return train, valid

def build_datasets(data_path: str, features_used: str):
    """Build train/valid/test lists + torchnet ListDataset with your existing cloud_loader."""
    pointcloud_train_files = glob(os.path.join(data_path, "train/*.txt"))
    pointcloud_test_files  = glob(os.path.join(data_path, "test/*.txt"))

    valid_list = np.random.choice(
        pointcloud_train_files, size=int(0.2 * len(pointcloud_train_files)), replace=False
    ).tolist()
    train_list = [p for p in pointcloud_train_files if p not in valid_list]
    test_list  = pointcloud_test_files

    train_set = tnt.dataset.ListDataset(train_list, functools.partial(cloud_loader, features_used=features_used))
    valid_set = tnt.dataset.ListDataset(valid_list, functools.partial(cloud_loader, features_used=features_used))
    test_set  = tnt.dataset.ListDataset(test_list,  functools.partial(cloud_loader, features_used=features_used))

    print(f"Number of training samples:   {len(train_list)}")
    print(f"Number of validation samples: {len(valid_list)}")
    print(f"Number of testing samples:    {len(test_list)}")
    return (train_list, valid_list, test_list), (train_set, valid_set, test_set)

def build_loaders(train_set, valid_set, batch_size: int):
    """Use your existing cloud_collate (unchanged)."""
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=cloud_collate
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=cloud_collate
    )
    return train_loader, valid_loader

def build_model_and_optimizer(args):
    """Construct PointNet and optimizer exactly as before."""
    model = PointNet(
        args.mlp1, args.mlp2, args.mlp3,
        n_class=args.n_classes,
        input_feat=args.n_input_feat,
        subsample_size=args.subsample_size,
        cuda=args.cuda
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=getattr(args, "milestones", [20, 40]), gamma=0.1)
    return model, optimizer, scheduler

# -------------------------
# 1) One training/validation step (logic preserved)
# -------------------------
def forward_batch(model, PCC, batch, device, ignore_index=None):
    """
    Your existing pattern:
      - PCC.run(model, clouds) returns [N_points, C] logits concatenated over the batch
      - labels are concatenated in cloud_collate -> shape [N_points]
    """
    clouds, labels = batch
    if device.type == "cuda":
        clouds = [c.to(device) for c in clouds]
        labels = labels.to(device)

    logits = PCC.run(model, clouds)           # [N, C]
    preds  = logits.argmax(dim=1)             # [N]

    # CrossEntropy expects [N, C] logits and [N] long labels
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index) if ignore_index is not None \
           else F.cross_entropy(logits, labels)

    # simple metrics (acc) at step level
    with torch.no_grad():
        valid_mask = labels != ignore_index if ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
        correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
        total   = valid_mask.sum().item()
        acc = correct / total if total > 0 else 0.0

    return loss, acc, total

def run_one_epoch(model, PCC, loader, optimizer, device, train: bool, ignore_index=None):
    """
    Train or validate for one pass over loader.
    Optimizer.step() only if train=True. Logic is unchanged from your flow.
    """
    model.train() if train else model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    it = tqdm(loader, desc="Train" if train else "Valid", leave=False)
    for batch in it:
        if train: optimizer.zero_grad(set_to_none=True)
        loss, acc, total = forward_batch(model, PCC, batch, device, ignore_index=ignore_index)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * total
        running_correct += int(acc * total)
        running_total += total
        it.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc  = running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc

# -------------------------
# 2) Full training loop
# -------------------------
def train_full(args):
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    # datasets & loaders (unchanged logic)
    (train_list, valid_list, test_list), (train_set, valid_set, _) = build_datasets(args.data_path, args.input_feats)
    train_loader, valid_loader = build_loaders(train_set, valid_set, args.batch_size)

    # model, optimizer, scheduler
    model, optimizer, scheduler = build_model_and_optimizer(args)
    PCC = PointCloudClassifier(args)  # keep your current classifier as-is

    best_valid_acc = -1.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, args.best_model_name)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(model, PCC, train_loader, optimizer, device, train=True, ignore_index=args.ignore_index)
        valid_loss, valid_acc = run_one_epoch(model, PCC, valid_loader, optimizer, device, train=False, ignore_index=args.ignore_index)
        scheduler.step()

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
              f"valid_loss={valid_loss:.4f} acc={valid_acc*100:.2f}% | "
              f"time={dt:.1f}s")

        # keep your original checkpoint criterion (e.g., best val acc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ Saved best model to: {ckpt_path} (val acc {best_valid_acc*100:.2f}%)")

    return ckpt_path

# -------------------------
# 3) Main 
# -------------------------
def build_args():
    """Mirror your existing hyper-parameters without changing defaults."""
    args = mock.Mock()
    # data/config
    args.data_path       = "data/"
    args.input_feats     = "xyzrgbi"
    args.n_input_feat    = len(args.input_feats)
    args.n_classes       = 4
    args.batch_size      = 8
    args.subsample_size  = 2048
    args.cuda            = 1  # set 0 to force CPU
    args.seed            = 42

    # model MLPs 
    args.mlp1 = [32, 32]
    args.mlp2 = [32, 64, 256]
    args.mlp3 = [128, 64, 32]

    # optimization
    args.lr              = 1e-3
    args.weight_decay    = 1e-4
    args.milestones      = [20, 40]
    args.epochs          = 50
    args.ignore_index    = None  # or 0 if you want to ignore "unclassified"

    # checkpointing
    args.checkpoint_dir  = "./"
    args.best_model_name = "best_model.pth"
    return args

def main():
    args = build_args()
    print("Starting training with the current (unchanged) logic...")
    ckpt = train_full(args)
    print(f"Training complete. Best checkpoint: {ckpt}")

# If this file is executed directly, run training.
if __name__ == "__main__":
    main()

