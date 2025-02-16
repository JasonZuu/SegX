from dataclasses import dataclass
import torch


@dataclass
class GradCamConfig:
    model_name = "resnet"
    norm_method = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = 0.3
    dataset = "ham" # "ham" or "chestx"

    log_dir = "log"
    interpret_dir = "log/grad_cam"


@dataclass
class GradientSHAPConfig:
    model_name = "resnet"
    norm_method = "zscore"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_shape = (3, 256, 256)
    nsample = 50
    alpha = 0.3
    dataset = "ham" # "ham" or "chestx"

    log_dir = "log"
    interpret_dir = "log/gradident_shap"


@dataclass
class SegXGradCamConfig:
    model_name = "resnet"
    norm_method = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = 0.3
    dataset = "ham" # "ham" or "chestx"

    log_dir = "log"
    interpret_dir = "log/seg_gradcam"


@dataclass
class SegXGradientSHAPConfig:
    model_name = "resnet"
    norm_method = "zscore"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_shape = (3, 256, 256)
    nsample = 50
    alpha = 0.3
    dataset = "ham" # "ham" or "chestx"

    log_dir = "log"
    interpret_dir = "log/seg_gradient_shap"