import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def train_transforms_data(batch: int, crop_size: int) -> torchvision.transforms.Compose:
    """
    Returns a torchvision.transforms.Compose object that transforms the data.
    """
    transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
    ])
    return transform

def train_image_foldering(data_dir: str, batch: int, size: int) -> torchvision.datasets.ImageFolder:
    """
    Returns a torchvision.datasets.ImageFolder object that contains the training data.
    """
    train_set = torchvision.datasets.ImageFolder(
        root='data/Gender/training', 
        transform=train_transforms_data(batch, size)
    )

    return train_set

def train_data_loader(train_set: torchvision.datasets.ImageFolder,) -> DataLoader:
    """
    Returns a torch.utils.data.DataLoader object that contains the data.
    """
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )
    return train_loader


