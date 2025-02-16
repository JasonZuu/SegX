from torchvision import transforms


def get_image_transform(image_size):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=means, std=stds)])
    return transformer


def get_mask_transform(image_size):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size, antialias=False),
        transforms.CenterCrop(image_size)])
    return transformer
    