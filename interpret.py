import torch
import os
import argparse
from pathlib import Path

from configs.interpret_configs import GradCamConfig, GradientSHAPConfig, SegXGradCamConfig, SegXGradientSHAPConfig
from configs.dataset_configs import HAMDatasetConfig, ChestXDatasetConfig
from datasets import HAMDataset, ChestXDataset
from algo_fn.gradcam_fn import grad_cam_fn
from algo_fn.shap_fn import gradient_shap_fn
from algo_fn.segx_gradcam import segx_grad_cam_fn
from algo_fn.segx_shap_fn import segx_gradient_shap_fn
from models import ResNetBased, DenseNetBased, UNet
from utils.misc import set_random_seed, get_log_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset', type=str, default='ham', choices=["ham", "chestx"], help='dataset')
    parser.add_argument("--algo", type=str, default="grad_cam", 
                        choices=["grad_cam", "gradient_shap", "segx_gradcam", 'segx_gradient_shap'], help="algo")
    parser.add_argument("--model", type=str, default="resnet",choices=["resnet", "densenet"])
    parser.add_argument("--log_dir", type=str, default="log", help="log_dir")
    parser.add_argument("--seed", type=int, help="seed", default=18087)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_random_seed(args.seed)

    # init dataset
    if args.dataset == "ham":
        dataset_config = HAMDatasetConfig()
        val_dataset = HAMDataset(dataset_config, set_type="val")
        test_dataset = HAMDataset(dataset_config, set_type="test", return_image_name=True)
    elif args.dataset == "chestx":
        dataset_config = ChestXDatasetConfig()
        val_dataset = ChestXDataset(dataset_config, set_type="val")
        test_dataset = ChestXDataset(dataset_config, set_type="test", return_image_name=True)
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")
    
    # init interpret config
    if args.algo == "grad_cam":
        interpret_config = GradCamConfig()
    elif args.algo == "gradient_shap":
        interpret_config = GradientSHAPConfig()
    elif args.algo == "segx_gradcam":
        interpret_config = SegXGradCamConfig()
    elif args.algo == "segx_gradient_shap":
        interpret_config = SegXGradientSHAPConfig()
    else:
        raise NotImplementedError(f"{args.algo} is not implemented")
    
    interpret_config.model_name = args.model
    interpret_config.device = args.device
    interpret_config.log_dir = os.path.join(args.log_dir, args.algo, f"{args.dataset}-{args.model}-{args.seed}")
    interpret_dir = os.path.join(interpret_config.interpret_dir, f"{args.dataset}-{args.model}-{args.seed}")
    interpret_config.interpret_dir = interpret_dir
    interpret_config.dataset = args.dataset
    Path(interpret_config.interpret_dir).mkdir(parents=True, exist_ok=True)

    # get state dict
    log_dir = get_log_dir(log_dir=args.log_dir, model=args.model, dataset=args.dataset, task="cls")
    interpret_config.log_dir = log_dir
    interpret_config.num_classes = dataset_config.num_classes

    weights_path = f"{log_dir}/model.pth"
    state_dict = torch.load(weights_path, weights_only=True, map_location=torch.device("cpu"))["model"]

    # init model
    if args.model == "resnet":
        model = ResNetBased(num_classes=interpret_config.num_classes, pretrained=False)
    elif args.model == "densenet":
        model = DenseNetBased(num_classes=interpret_config.num_classes, pretrained=False)
    else:
        raise NotImplementedError(f"{args.model} is not implemented")
    
    model.load_state_dict(state_dict)
    model.to(interpret_config.device)
    model.eval()

    #load segmentation model for segx
    if args.algo.startswith("segx"):
        seg_model = UNet(out_channels=1)
        seg_log_dir = get_log_dir(log_dir=args.log_dir, model='unet', dataset=args.dataset, task="seg")
        weight_path = f"{seg_log_dir}/model.pth"
        state_dict = torch.load(weight_path, weights_only=True)["model"]
        seg_model.load_state_dict(state_dict)
        seg_model.to(interpret_config.device)

    # cam
    if args.algo == "grad_cam":
        grad_cam_fn(model, val_dataset, test_dataset, interpret_config)
    elif args.algo == "gradient_shap":
        gradient_shap_fn(model, val_dataset, test_dataset, interpret_config)
    elif args.algo == "segx_gradcam":
        segx_grad_cam_fn(model, seg_model, val_dataset, test_dataset, interpret_config)
    elif args.algo == "segx_gradient_shap":
        segx_gradient_shap_fn(model, seg_model, val_dataset, test_dataset, interpret_config)
    else:
        raise NotImplementedError(f"{args.algo} is not implemented")
