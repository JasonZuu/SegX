import albumentations as A
import numpy as np
import cv2


def get_shift_transform():
    return A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=0, scale_limit=0, p=1)

def get_rotate_transform():
    return A.ShiftScaleRotate(shift_limit=0, rotate_limit=30, scale_limit=0, p=1)

def get_scale_transform():
    return A.ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=0.1, p=1)

def get_horizontal_flip_transform():
    return A.HorizontalFlip(p=1.0)

def get_vertical_flip_transform():
    return A.VerticalFlip(p=1.0)

def get_brightness_transform():
    return A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1)

def get_contrast_transform():
    return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=1)

def get_blur_transform():
    return A.GaussianBlur(blur_limit=(3, 7), p=1)

def get_noise_transform():
    return A.GaussNoise(var_limit=(100.0, 200.0), p=1)


class RandomMaskApply:
    def __init__(self, image_shape, dropout_ratio=0.6):
        self.image_shape = image_shape
        self.mask = np.random.rand(image_shape[0], image_shape[1]) > dropout_ratio
        self.mask = self.mask.astype(np.uint8)

    def _resize_image(self, image):
        return cv2.resize(image, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_AREA)

    def __call__(self, image):
        _image = self._resize_image(image)
        _image = _image * self.mask[..., np.newaxis]  # 确保掩码对每个通道有效
        return {"image": _image}
    

def get_random_mask_transform(image_shape, dropout_ratio=0.6):
    return RandomMaskApply(image_shape, dropout_ratio=dropout_ratio)
