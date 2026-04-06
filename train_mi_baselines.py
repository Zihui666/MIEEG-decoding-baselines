from pathlib import Path
import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Unified BCIC IV-2a baseline training script.
#
# This file provides a unified PyTorch benchmark framework for multiple
# MI-EEG backbones, including:
# - EEGNet
# - DeepConvNet
# - TCANet
# - EEG-Conformer
#
# Attribution:
# - The strict preprocessing / training protocol used for the TCANet-style
#   setting is adapted from the public TCANet implementation for BCIC IV-2a.
# - EEGNet is based on the architecture proposed by Lawhern et al. (2018).
# - DeepConvNet is based on the architecture proposed by Schirrmeister et al. (2017).
# - EEG-Conformer is included here as an additional baseline backbone.
#
# The benchmark framework itself, CLI, result export utilities, and stricter
# checkpoint-selection logic were reorganized and extended in this repository.


# =========================================================
# Repro helpers
# =========================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_subject_number(subject: str) -> int:
    digits = "".join(ch for ch in subject if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse subject number from {subject}")
    return int(digits)


# =========================================================
# Models
# =========================================================
class EEGNet(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_classes: int,
        n_times: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        feat_dim = self._infer_feat_dim(n_chans=n_chans, n_times=n_times)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _infer_feat_dim(self, n_chans: int, n_times: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chans, n_times)
            x = self.block1(x)
            x = self.block2(x)
            return x.flatten(start_dim=1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        features = x.flatten(start_dim=1)
        logits = self.classifier(features)
        return logits, features


class DeepConvNet(nn.Module):
    def __init__(self, n_chans: int, n_classes: int, n_times: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(25, 25, kernel_size=(n_chans, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
            nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
            nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
            nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
        )
        feat_dim = self._infer_feat_dim(n_chans=n_chans, n_times=n_times)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _infer_feat_dim(self, n_chans: int, n_times: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chans, n_times)
            x = self.features(x)
            return x.flatten(start_dim=1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        features = x.flatten(start_dim=1)
        logits = self.classifier(features)
        return logits, features


class MSCNet(nn.Module):
    def __init__(self, f1=16, pooling_size=56, dropout_rate=0.5, number_channel=22):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 125), stride=(1, 1), padding="same"),
            nn.Conv2d(f1, f1, (number_channel, 1), stride=(1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)),
            nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 62), stride=(1, 1), padding="same"),
            nn.Conv2d(f1, f1, (number_channel, 1), stride=(1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)),
            nn.Dropout(dropout_rate),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 31), stride=(1, 1), padding="same"),
            nn.Conv2d(f1, f1, (number_channel, 1), stride=(1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.squeeze(2).transpose(1, 2)
        return x


class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (padding, 0))
        return super().forward(x)


class TCNBlock(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, drop_prob):
        super().__init__()
        self.activation = nn.ELU()
        self.layers = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, kernel_size=1, bias=False) if input_dimension != filters else None

        for i in range(depth):
            dilation = 2 ** i
            block = nn.Sequential(
                CausalConv1d(
                    in_channels=input_dimension if i == 0 else filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                nn.ELU(),
                nn.Dropout(drop_prob),
                CausalConv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                nn.ELU(),
                nn.Dropout(drop_prob),
            )
            self.layers.append(block)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        res = x if self.downsample is None else self.downsample(x)
        for layer in self.layers:
            out = layer(x)
            out = self.activation(out + res)
            res = out
            x = out
        return out.permute(0, 2, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})")
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        bsz, n_tokens, emb = x.shape
        heads = self.num_heads
        head_dim = emb // heads
        q = self.queries(x).view(bsz, n_tokens, heads, head_dim).transpose(1, 2)
        k = self.keys(x).view(bsz, n_tokens, heads, head_dim).transpose(1, 2)
        v = self.values(x).view(bsz, n_tokens, heads, head_dim).transpose(1, 2)
        energy = torch.matmul(q, k.transpose(-2, -1))
        att = torch.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, emb)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x):
        res = self.fn(x)
        return self.layernorm(self.drop(res) + x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=2, drop_p=0.5):
        super().__init__()
        self.block = ResidualAdd(MultiHeadAttention(emb_size, num_heads, drop_p), emb_size, drop_p)

    def forward(self, x):
        return self.block(x)


class TransformerEncoder(nn.Module):
    def __init__(self, heads, depth, emb_size, drop_p=0.5):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(emb_size, heads, drop_p) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TCANet(nn.Module):
    def __init__(
        self,
        n_chans=22,
        n_classes=4,
        n_times=1000,
        f1=16,
        pooling_size=56,
        drop_prob=0.5,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_filters=16,
        heads=2,
        attn_depth=6,
        max_norm_const=0.25,
    ):
        super().__init__()
        self.max_norm_const = max_norm_const
        self.mscnet = MSCNet(f1=f1, pooling_size=pooling_size, dropout_rate=drop_prob, number_channel=n_chans)
        self.tcn_block = TCNBlock(input_dimension=f1 * 3, depth=tcn_depth, kernel_size=tcn_kernel_size, filters=tcn_filters, drop_prob=0.25)
        self.sa = TransformerEncoder(heads=heads, depth=attn_depth, emb_size=tcn_filters, drop_p=0.25)
        self.drop = nn.Dropout(0.25)
        feat_dim = self._infer_feat_dim(n_chans=n_chans, n_times=n_times)
        self.classifier = nn.Linear(feat_dim, n_classes)
        self.classifier.register_forward_pre_hook(self.apply_max_norm_classifier)

    def _infer_feat_dim(self, n_chans: int, n_times: int):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chans, n_times)
            x = self.mscnet(x)
            x = self.tcn_block(x)
            x = self.drop(self.sa(x) + x)
            return x.flatten(start_dim=1).shape[1]

    def apply_max_norm_classifier(self, module, inputs):
        with torch.no_grad():
            norm = self.classifier.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm_const)
            self.classifier.weight.data *= desired / (norm + 1e-8)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mscnet(x)
        x = self.tcn_block(x)
        x = self.drop(self.sa(x) + x)
        features = x.flatten(start_dim=1)
        logits = self.classifier(features)
        return logits, features


# EEG Conformer blocks
class EEGCResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class EEGCFeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class EEGCPatchEmbedding(nn.Module):
    def __init__(
        self,
        n_filters_time: int,
        filter_time_length: int,
        n_channels: int,
        pool_time_length: int,
        stride_avg_pool: int,
        drop_prob: float,
    ):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)),
            nn.Dropout(p=drop_prob),
        )
        self.projection = nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        x = x.squeeze(2).transpose(1, 2)
        return x


class EEGCTransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, att_drop: float, forward_expansion: int = 4):
        super().__init__()
        self.attn = EEGCResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, att_drop),
                nn.Dropout(att_drop),
            )
        )
        self.ffn = EEGCResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                EEGCFeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=att_drop),
                nn.Dropout(att_drop),
            )
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class EEGCTransformer(nn.Module):
    def __init__(self, num_layers: int, emb_size: int, num_heads: int, att_drop: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EEGCTransformerEncoderBlock(emb_size, num_heads, att_drop) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EEGCClassifierHead(nn.Module):
    def __init__(self, final_fc_length: int, n_classes: int, hidden_channels: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, hidden_channels),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.final = nn.Linear(hidden_channels, n_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        features = self.fc(x)
        logits = self.final(features)
        return logits, features


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_classes: int,
        n_times: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        num_layers: int = 6,
        num_heads: int = 10,
        att_drop_prob: float = 0.5,
    ):
        super().__init__()
        self.patch_embedding = EEGCPatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
        )
        self.transformer = EEGCTransformer(
            num_layers=num_layers,
            emb_size=n_filters_time,
            num_heads=num_heads,
            att_drop=att_drop_prob,
        )
        final_fc_length = self._infer_fc_size(n_chans=n_chans, n_times=n_times)
        self.classifier = EEGCClassifierHead(final_fc_length=final_fc_length, n_classes=n_classes)

    def _infer_fc_size(self, n_chans: int, n_times: int) -> int:
        with torch.no_grad():
            x = torch.ones((1, 1, n_chans, n_times))
            x = self.patch_embedding(x)
            return x.shape[1] * x.shape[2]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        logits, features = self.classifier(x)
        return logits, features


# =========================================================
# Training helpers
# =========================================================
@dataclass
class TrainConfig:
    model: str = "tcanet"
    batch_size: int = 72
    epochs: int = 1000
    lr: float = 1e-3
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 0
    number_aug: int = 1
    number_seg: int = 8
    use_interaug: bool = True
    save_dir: str = "runs_mi_strict_baselines"
    seed_offset: int = 1234


def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    return x, y


def standardize_like_official(train_x: np.ndarray, test_x: np.ndarray):
    mean = float(train_x.mean())
    std = float(train_x.std())
    std = std if std > 1e-8 else 1.0
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def split_train_val_like_official(x: np.ndarray, y: np.ndarray, subject_number: int, n_classes: int = 4):
    seed = 1234 + subject_number
    train_data_list, train_label_list = [], []
    val_data_list, val_label_list = [], []

    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        cls_x = x[cls_idx]
        cls_y = y[cls_idx]
        n_samples = len(cls_x)
        n_val = n_samples // 5
        indices = np.arange(n_samples)
        np.random.seed(seed + cls)
        index_shuffled = np.random.permutation(indices)
        index_val = index_shuffled[:n_val]
        index_train = index_shuffled[n_val:]
        train_data_list.append(cls_x[index_train])
        train_label_list.append(cls_y[index_train])
        val_data_list.append(cls_x[index_val])
        val_label_list.append(cls_y[index_val])

    tr_x = np.concatenate(train_data_list, axis=0)
    tr_y = np.concatenate(train_label_list, axis=0)
    val_x = np.concatenate(val_data_list, axis=0)
    val_y = np.concatenate(val_label_list, axis=0)
    return tr_x, tr_y, val_x, val_y


def interaug_like_official(all_x: np.ndarray, all_y: np.ndarray, batch_size: int, n_classes: int, number_aug: int, number_seg: int):
    aug_data = []
    aug_label = []
    records_per_class = number_aug * int(batch_size / n_classes)
    seg_points = all_x.shape[-1] // number_seg

    for cls in range(n_classes):
        cls_idx = np.where(all_y == cls)[0]
        tmp_data = all_x[cls_idx]
        if tmp_data.shape[0] == 0:
            continue
        tmp_aug = np.zeros((records_per_class, all_x.shape[1], all_x.shape[2]), dtype=np.float32)
        for rec_idx in range(records_per_class):
            for seg_idx in range(number_seg):
                rand_idx = np.random.randint(0, tmp_data.shape[0], number_seg)
                start = seg_idx * seg_points
                end = (seg_idx + 1) * seg_points if seg_idx < number_seg - 1 else all_x.shape[-1]
                tmp_aug[rec_idx, :, start:end] = tmp_data[rand_idx[seg_idx], :, start:end]
        aug_data.append(tmp_aug)
        aug_label.append(np.full((records_per_class,), cls, dtype=np.int64))

    aug_x = np.concatenate(aug_data, axis=0)
    aug_y = np.concatenate(aug_label, axis=0)
    shuffle = np.random.permutation(len(aug_y))
    return aug_x[shuffle], aug_y[shuffle]


def build_model(model_name: str, n_chans: int, n_classes: int, n_times: int) -> nn.Module:
    name = model_name.lower()
    if name == "eegnet":
        return EEGNet(n_chans=n_chans, n_classes=n_classes, n_times=n_times)
    if name == "deepconvnet":
        return DeepConvNet(n_chans=n_chans, n_classes=n_classes, n_times=n_times)
    if name == "tcanet":
        return TCANet(n_chans=n_chans, n_classes=n_classes, n_times=n_times)
    if name == "eegconformer":
        return EEGConformer(n_chans=n_chans, n_classes=n_classes, n_times=n_times)
    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = float((np.array(preds_all) == np.array(labels_all)).mean())
    return avg_loss, acc, np.array(labels_all), np.array(preds_all)


def save_subject_outputs(save_root: Path, result: Dict):
    subject = result["subject"]
    history_path = save_root / f"{subject}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss_last_batch", "train_acc_last_batch", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(result["history"])

    preds_path = save_root / f"{subject}_predictions.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for y_true, y_pred in zip(result["y_true"], result["y_pred"]):
            writer.writerow([int(y_true), int(y_pred)])

    ckpt_path = save_root / f"{result['model_name']}_{subject}.pt"
    torch.save(
        {
            "model_name": result["model_name"],
            "subject": subject,
            "best_epoch": result["best_epoch"],
            "best_val_acc": result["best_val_acc"],
            "best_val_loss": result["best_val_loss"],
            "test_acc": result["test_acc"],
            "test_loss": result["test_loss"],
            "history": result["history"],
            "state_dict": result["best_state"],
        },
        ckpt_path,
    )


def save_summary(save_root: Path, cfg: TrainConfig, per_subject_results: List[Dict]):
    summary_csv = save_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "best_epoch", "best_val_acc", "best_val_loss", "test_acc", "test_loss"],
        )
        writer.writeheader()
        for result in per_subject_results:
            writer.writerow(
                {
                    "subject": result["subject"],
                    "best_epoch": result["best_epoch"],
                    "best_val_acc": result["best_val_acc"],
                    "best_val_loss": result["best_val_loss"],
                    "test_acc": result["test_acc"],
                    "test_loss": result["test_loss"],
                }
            )

    test_accs = [float(r["test_acc"]) for r in per_subject_results]
    aggregate = {
        "model": cfg.model,
        "mean_test_acc": float(np.mean(test_accs)) if test_accs else None,
        "std_test_acc": float(np.std(test_accs)) if test_accs else None,
        "subjects": [r["subject"] for r in per_subject_results],
        "config": asdict(cfg),
    }
    with (save_root / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)



def train_subject(subject: str, train_npz: Path, test_npz: Path, cfg: TrainConfig, device: torch.device):
    subject_number = parse_subject_number(subject)
    set_global_seed(cfg.seed_offset + subject_number)

    all_x, all_y = load_npz(train_npz)
    test_x, test_y = load_npz(test_npz)

    all_x, test_x, _, _ = standardize_like_official(all_x, test_x)

    shuffle_num = np.random.permutation(len(all_x))
    all_x = all_x[shuffle_num]
    all_y = all_y[shuffle_num]

    tr_x, tr_y, val_x, val_y = split_train_val_like_official(all_x, all_y, subject_number=subject_number, n_classes=len(np.unique(all_y)))

    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_x), torch.from_numpy(tr_y)), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    n_chans = tr_x.shape[1]
    n_times = tr_x.shape[2]
    n_classes = len(np.unique(all_y))

    model = build_model(cfg.model, n_chans=n_chans, n_classes=n_classes, n_times=n_times).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    best_epoch = -1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        last_train_loss = None
        last_train_acc = None
        for x, y in train_loader:
            if cfg.use_interaug and cfg.number_aug > 0:
                aug_x, aug_y = interaug_like_official(
                    all_x,
                    all_y,
                    batch_size=x.size(0),
                    n_classes=n_classes,
                    number_aug=cfg.number_aug,
                    number_seg=cfg.number_seg,
                )
                x = torch.cat([x, torch.from_numpy(aug_x)], dim=0)
                y = torch.cat([y, torch.from_numpy(aug_y)], dim=0)

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            last_train_acc = float((preds == y).float().mean().item())
            last_train_loss = float(loss.item())

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss_last_batch": float(last_train_loss),
                "train_acc_last_batch": float(last_train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            test_loss_now, test_acc_now, _, _ = evaluate(model, test_loader, criterion, device)
            print(
                f"{cfg.model}:{subject} epoch={epoch} train_acc(last_batch)={last_train_acc:.4f} "
                f"train_loss(last_batch)={last_train_loss:.6f} val_acc={val_acc:.6f} "
                f"val_loss={val_loss:.7f} test_acc={test_acc_now:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    return {
        "model_name": cfg.model,
        "subject": subject,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "history": history,
        "best_state": best_state,
    }


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Strict-ish baseline training for BCIC IV-2a")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing A01_train.npz ... A09_test.npz")
    parser.add_argument("--model", type=str, default="tcanet", choices=["eegnet", "deepconvnet", "tcanet", "eegconformer"])
    parser.add_argument("--subjects", type=str, default="A01,A02,A03,A04,A05,A06,A07,A08,A09")
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--number_aug", type=int, default=1)
    parser.add_argument("--number_seg", type=int, default=8)
    parser.add_argument("--disable_interaug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="runs_mi_strict_baselines")
    args = parser.parse_args()

    cfg = TrainConfig(
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        number_aug=args.number_aug,
        number_seg=args.number_seg,
        use_interaug=not args.disable_interaug,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    save_root = Path(cfg.save_dir) / cfg.model
    save_root.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    print("Using device:", device)
    print("Model:", cfg.model)
    print("Subjects:", subjects)
    print("batch_size:", cfg.batch_size, "epochs:", cfg.epochs, "lr:", cfg.lr, "betas:", (cfg.beta1, cfg.beta2))
    print("use_interaug:", cfg.use_interaug, "number_aug:", cfg.number_aug, "number_seg:", cfg.number_seg)

    per_subject_results: List[Dict] = []
    for subject in subjects:
        train_npz = data_dir / f"{subject}_train.npz"
        test_npz = data_dir / f"{subject}_test.npz"
        if not train_npz.exists() or not test_npz.exists():
            raise FileNotFoundError(f"Missing files for {subject}: {train_npz.name} / {test_npz.name}")

        result = train_subject(subject, train_npz, test_npz, cfg, device)
        per_subject_results.append(result)
        save_subject_outputs(save_root, result)
        print(
            f"{cfg.model}:{subject} | best_epoch={result['best_epoch']} | "
            f"best_val_acc={result['best_val_acc']:.4f} | test_acc={result['test_acc']:.4f}"
        )

    save_summary(save_root, cfg, per_subject_results)
    test_accs = [float(r["test_acc"]) for r in per_subject_results]
    mean_acc = float(np.mean(test_accs)) if test_accs else float("nan")
    std_acc = float(np.std(test_accs)) if test_accs else float("nan")
    print("\n===== Summary =====")
    print(f"{cfg.model}: mean_test_acc={mean_acc:.4f}, std_test_acc={std_acc:.4f}")
    print(f"Saved outputs to: {save_root}")


if __name__ == "__main__":
    main()
