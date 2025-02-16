from dataclasses import dataclass
import torch


@dataclass
class ClassificationConfig:
    seed = 0
    epoch_num = 100
    num_classes = 1
    early_stop_epochs = 10
    used_ratio = 1.0  # dataset used ratio for training
    cutoff = None
    warmup_epochs = 10
    lr_decay_gamma = 0.95

    # search space
    model_lr = 1e-4
    batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = "log"


@dataclass
class SegmentationConfig:
    seed = 0
    epoch_num = 100
    num_classes = 1
    early_stop_epochs = 10
    used_ratio = 1.0  # dataset used ratio for training
    cutoff = None
    warmup_epochs = 1
    lr_decay_gamma = 0.95

    # search space
    model_lr = 1e-3
    batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = "log"
