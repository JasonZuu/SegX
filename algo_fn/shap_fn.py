import os
import torch
import shap
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path

from configs.interpret_configs import GradientSHAPConfig
from utils.misc import unnormalize, get_saliency_map, get_overlayed_image
from datasets import chestx_label_to_idx, ham_label_to_idx
from datasets.loader import get_dataloader
from algo_fn.test_fn import _cls_test_loop


def gradient_shap_fn(model, val_dataset, test_dataset, config: GradientSHAPConfig):
    """
    Computes SHAP values for each image in the dataset using Gradient SHAP and saves the SHAP images,
    input images, and overlaid images to the specified directories within the log directory.

    Parameters:
    - model: Trained PyTorch model.
    - dataset: PyTorch dataset containing images.
    - config: Configuration object containing directory and image shape info.
    """
    if config.dataset == "ham":
        label_to_idx = ham_label_to_idx
    elif config.dataset == "chestx":
        label_to_idx = chestx_label_to_idx
    else:
        raise NotImplementedError(f"{config.dataset} is not implemented")
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Set the model to evaluation mode
    model.eval()

    # get the cutoff on the val_dataset
    val_loader = get_dataloader(val_dataset, mode="original",
                                batch_size=64,
                                num_classes=config.num_classes)
    val_metric_dict = _cls_test_loop(model, loader=val_loader, device=config.device, cutoff=None, run_bootstrap=False)
    cutoff = val_metric_dict["cutoff"]
    
    # Ensure the output directories exist
    raw_interpret_dir = os.path.join(config.interpret_dir, "raw_interpret_map")
    saliency_dir = os.path.join(config.interpret_dir, "saliency_map")
    image_dir = os.path.join(config.interpret_dir, "image")
    overlaid_dir = os.path.join(config.interpret_dir, "overlaid")
    for directory in [saliency_dir, image_dir, overlaid_dir, raw_interpret_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    #take val_dataset images as background
    background_images = [val_dataset[i][0] for i in range(config.nsample)]
    background_images = torch.stack(background_images).to(config.device)

    # Initialize SHAP's DeepExplainer
    explainer = shap.GradientExplainer(model, background_images)
    
    # Iterate through each image in the test_dataset
    pbar = tqdm(total=len(test_dataset), desc="Generate Gradient SHAP map")
    for idx, (image, label, image_name) in enumerate(test_dataset):
        image_save_path = os.path.join(image_dir, f"{image_name}.png")
        if os.path.exists(image_save_path):
            pbar.update(1)
            continue
        label_category = torch.where(label == 1)[0].numpy()
        original_img = unnormalize(image)
        save_image(original_img, image_save_path)

        image = image.unsqueeze(0).to(config.device)
        
        shap_map_dict = get_shap_map_dict(image, model, cutoff, explainer, config.nsample)

        for idx, shap_map in shap_map_dict.items():
            saliency_map = get_saliency_map(shap_map)
            overlayed_image = get_overlayed_image(original_img, saliency_map, alpha=config.alpha)

            disease_label = idx_to_label[idx]
            pred_label = 'True Positive' if idx in label_category else 'False Positive'

            # Save the saliency map and overlayed image
            save_image(saliency_map, f"{saliency_dir}/{image_name}_{disease_label}_{pred_label}.png")
            save_image(overlayed_image, f"{overlaid_dir}/{image_name}_{disease_label}_{pred_label}.png")
            np.save(f"{raw_interpret_dir}/{image_name}_{disease_label}_{pred_label}.npy", shap_map)
        pbar.update(1)

    pbar.close()


def get_shap_map_dict(image, model, cutoff, explainer, nsample):
    """
    Compute SHAP values for the given image using the provided model and explainer.

    Parameters:
    - image: Image tensor.
    - model: Trained PyTorch model.
    - cutoff: Cutoff value for binary classification.
    - explainer: SHAP explainer object.

    Returns:
    - shap_maps: SHAP values for the image.
    """
    shap_map_dict = {}

    y_pred = model(image, use_sigmoid=True)
    y_pred = (y_pred.cpu().detach().numpy() > cutoff).astype(int)

    # Compute SHAP values
    shap_values = explainer.shap_values(image, nsamples=nsample)
    shap_values = shap_values.squeeze(0).transpose(3, 0, 1, 2) # (C, H, W, Class) -> (Class, C, H, W)
    shap_maps = np.mean(np.abs(shap_values), axis=1)  # Take the absolute value of the SHAP values as importance

    # store the positive class shap values
    for idx in range(y_pred.shape[1]):
        if y_pred[0, idx] == 1:
            shap_map_dict[idx] = shap_maps[idx]
    
    return shap_map_dict