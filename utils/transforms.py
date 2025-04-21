import torch
import torchvision
from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # 将图像转换为张量
        image = F.to_tensor(image)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 调整图像大小
        image = F.resize(image, self.size)
        return image, target

def get_transform(train=True):
    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms.append(Resize((320, 320)))  # SSD Lite使用320x320
    
    return Compose(transforms)
