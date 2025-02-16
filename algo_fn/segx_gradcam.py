import torch
from torchcam.methods import GradCAM
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np

from configs.interpret_configs import GradCamConfig
from utils.misc import unnormalize, get_saliency_map, get_overlayed_image
from algo_fn.test_fn import _cls_test_loop
from datasets.loader import get_dataloader
from datasets import chestx_label_to_idx, ham_label_to_idx


def segx_grad_cam_fn(model: torch.nn.Module, 
                     seg_model: torch.nn.Module,
                     val_dataset,
                     test_dataset,
                     config: GradCamConfig):
    if config.dataset == "ham":
        label_to_idx = ham_label_to_idx
    elif config.dataset == "chestx":
        label_to_idx = chestx_label_to_idx
    else:
        raise NotImplementedError(f"{config.dataset} is not implemented")
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model.eval()
    seg_model.eval()

    image_shape = test_dataset.get_image_shape()
    cam_transform = transforms.Resize(image_shape, antialias=True)

    raw_interpret_dir = os.path.join(config.interpret_dir, "raw_interpret_map")
    saliency_dir = os.path.join(config.interpret_dir, "saliency_map")
    image_dir = os.path.join(config.interpret_dir, "image")
    overlayed_dir = os.path.join(config.interpret_dir, "overlayed")
    for directory in [saliency_dir, image_dir, overlayed_dir, raw_interpret_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # get the cutoff on the val_dataset
    val_loader = get_dataloader(val_dataset, mode="original",
                                batch_size=64,
                                num_classes=config.num_classes)
    val_metric_dict = _cls_test_loop(model, loader=val_loader, device=config.device, cutoff=None, run_bootstrap=False)
    cutoff = val_metric_dict["cutoff"]

    # get the last convolutional layer
    last_conv_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = name

    cam_extractor = GradCAM(model, target_layer=last_conv_layer, input_shape=image_shape)

    pbar = tqdm(total=len(test_dataset), desc="Generate Grad-CAM map")

    for idx, (image, label, image_name) in enumerate(test_dataset):
        label_category = torch.where(label == 1)[0].numpy()
        with torch.no_grad():
            image = image.to(config.device).unsqueeze(0)
            seg_mask = seg_model(image)
            seg_mask = (seg_mask > 0.5)
            seg_mask = seg_mask.cpu().detach().numpy()
        act_map_dict = _get_grad_cam(image, model, cutoff, cam_extractor, cam_transform)

        if len(act_map_dict) == 0:
            pbar.update(1)
            continue

        original_img = unnormalize(image.squeeze(0)).to("cpu")
        save_image(original_img, f"{image_dir}/{image_name}.png")

        # get the saliency map and overlayed image
        for idx, act_map in act_map_dict.items():
            act_map_np = act_map.cpu().detach().numpy()
            act_map_np = act_map_np * seg_mask
            saliency_map = get_saliency_map(act_map_np, top_n_percent=5)
            overlayed_img = get_overlayed_image(original_img, saliency_map, alpha=config.alpha)

            disease_label = idx_to_label[idx]
            pred_label = 'True Positive' if idx in label_category else 'False Positive'

            save_image(saliency_map, f"{saliency_dir}/{image_name}_{disease_label}_{pred_label}.png")
            save_image(overlayed_img, f"{overlayed_dir}/{image_name}_{disease_label}_{pred_label}.png")
            np.save(f"{raw_interpret_dir}/{image_name}_{disease_label}_{pred_label}.npy", act_map_np)
        pbar.update(1)
    pbar.close()
       

def _get_grad_cam(image, model, cutoff, cam_extractor, cam_transformer):
    y_score = model(image, use_sigmoid=True)
    y_pred = (y_score.cpu().detach().numpy() > cutoff).astype(int)

    act_map_dict = {}
    for idx in range(y_pred.shape[1]):
        if y_pred[0, idx] == 1:
            act_map = cam_extractor(idx, y_score.clone(), retain_graph=True)[0]
            act_map = cam_transformer(act_map).squeeze(0)
            act_map_dict[idx] = act_map
    return act_map_dict

