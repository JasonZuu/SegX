import torch
import copy
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import os


def sample_state_dict(state_dict, num_samples:int, module_prefix:str='classify_module', random_noise=0.1):
    """
    Sample parameters of a specific module in the model.

    Args:
    state_dict -- state dictionary of the model
    num_samples -- number of samples to be generated
    module_prefix -- prefix of the model module to be sampled
    random_noise -- range of the uniform distribution to sample from, centered around the original parameter value

    Returns:
    sampled_state_dicts -- list of sampled state dictionaries
    """
    sampled_state_dicts = []

    for i in range(num_samples):
        # Create a copy of the current model parameters
        sampled_state_dict = copy.deepcopy(state_dict)
        
        # Sample only the parameters of the specified module
        for key in sampled_state_dict:
            if key.startswith(module_prefix):
                param_value = sampled_state_dict[key]
                sampled_state_dict[key] = param_value + (torch.rand(1) * 2 - 1)*random_noise*param_value

        sampled_state_dicts.append(sampled_state_dict)

    return sampled_state_dicts


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.classify_module = nn.Linear(10, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classify_module(x)
        return x

def test_sampled_models():
    model = SimpleNet()
    num_samples = 10
    original_state_dict = model.state_dict()
    module_prefix = 'classify_module'
    random_noise = 0.1

    sampled_state_dicts = sample_state_dict(original_state_dict, num_samples, module_prefix, random_noise)
    print(f"Sampled {num_samples} state dictionaries from the original state dictionary.")


def set_grad_flag(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def unnormalize(image: torch.Tensor):
    """
    Unnormalize a tensor image with mean and std deviation.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for t, m, s in zip(image, means, stds):
        t.mul_(s).add_(m)
    return image



def set_random_seed(seed):
    """
    Set the random number seed for Python, NumPy, and PyTorch.

    Parameters:
    seed (int): The seed value to be set for all random number generators.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator
    torch.manual_seed(seed)

    # Additionally for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def random_split_dataset(*args, set_type='train', seed=None):
    """
    Splits the datasets into train, validation, and test sets with proportions 8:1:1,
    and returns the specified set (train, val, or test) for each dataset.

    Parameters:
    *args (array-like): Variable number of array-like datasets (e.g., image_paths, sensitives, labels).
    set_type (str): The dataset to return. One of 'train', 'val', or 'test'.

    Returns:
    Specified split of each dataset in the order they were passed.
    """
    # Set the random seed
    if seed is not None:
        np.random.seed(seed)

    # Total number of examples in the first dataset
    total_examples = len(args[0])

    # Creating indices and shuffling them
    indices = np.arange(total_examples)
    np.random.shuffle(indices)

    # Calculate split sizes
    train_end = int(total_examples * 0.7)
    val_end = train_end + int(total_examples * 0.1)

    # Split the indices
    if set_type == 'train':
        selected_indices = indices[:train_end]
    elif set_type == 'val':
        selected_indices = indices[train_end:val_end]
    elif set_type == 'test':
        selected_indices = indices[val_end:]
    elif set_type == 'all':
        selected_indices = indices
    else:
        raise ValueError("Invalid value for set_type. Choose 'train', 'val', 'test', or 'all'.")

    # Split the data and return
    selected_indices = selected_indices.astype(int)
    return [data[selected_indices] for data in args]


def get_saliency_map(act_map, top_n_percent=5, norm_method=None):
    # Normalize the act_map based on the specified method
    if norm_method == "minmax":
        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())
    elif norm_method == "mean":
        act_map = (act_map - act_map.mean()) / (act_map.max() - act_map.min())
    elif norm_method == "zscore":
        act_map = (act_map - act_map.mean()) / act_map.std()
    else:
        pass

    # Convert to numpy if it's a tensor
    if torch.is_tensor(act_map):
        act_map = act_map.cpu().numpy()

    # Flatten the act_map to calculate the threshold
    flattened_map = act_map.flatten()
    
    # Get the threshold for the top n% values
    threshold = np.percentile(flattened_map, 100 - top_n_percent)

    # Create a binary map: 1 for the top n% values, 0 otherwise
    binary_map = np.where(act_map > threshold, 1, 0)

    return torch.tensor(binary_map).float()


def get_overlayed_image(original_img, mask, alpha=0.3):
    """
    Overlay the binary mask on the original image.

    Parameters:
    original_img (torch.tensor): The original image.
    mask (torch.tensor): The binary mask.
    alpha (float): The alpha value for the overlay.

    Returns:
    overlayed_img (torch.tensor): The overlayed image.
    """
        # Ensure mask is in the correct shape (H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # Add a channel dimension to make it (1, H, W)
    if mask.dim() == 4:
        mask = mask.squeeze(0)

    # Repeat mask to match the number of channels of the original image (C, H, W)
    mask = mask.repeat(original_img.size(0), 1, 1)  # Repeat mask to match the image channels

    # Ensure the original image values are between 0 and 1 for proper blending
    if original_img.max() > 1:
        original_img = original_img / 255.0

    # Create the overlay by blending the original image and the mask
    overlayed_img = original_img * (1 - alpha) + mask * alpha

    return overlayed_img



def plot_mean_and_std(mean_gradcam, std_gradcam, save_fpath_mean=None, save_fpath_std=None,
                      colormap_name='jet'):
    fig_mean, ax_mean = plt.subplots()
    ax_mean.imshow(mean_gradcam, cmap=colormap_name)
    ax_mean.axis('off')  # Turn off axes
    plt.tight_layout()  # Use tight layout

    if save_fpath_mean:
        plt.savefig(save_fpath_mean, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close(fig_mean)
    
    # Plot std surface as an image
    fig_std, ax_std = plt.subplots()
    ax_std.imshow(std_gradcam, cmap=colormap_name)
    ax_std.axis('off')  # Turn off axes
    plt.tight_layout()  # Use tight layout

    if save_fpath_std:
        plt.savefig(save_fpath_std, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close(fig_std)


def category_to_onehot(category_labels, num_classes):
    """
    Convert a category to a one-hot encoded tensor.

    Parameters:
    category_labels (np.array): labels in category format.
    num_classes (int): The number of classes in the dataset.

    Returns:
    onehot (torch.Tensor): The one-hot encoded tensor.
    """
    onehot = torch.zeros((len(category_labels), num_classes))
    for i, label in enumerate(category_labels):
        onehot[i, label] = 1
    return onehot


def get_log_dir(log_dir, dataset, task, model):
    log_dir = os.path.join(log_dir, f"{dataset}-{task}", model)
    return log_dir