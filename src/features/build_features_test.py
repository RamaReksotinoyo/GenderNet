import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def test_transforms_data(batch: int, crop_size: int) -> torchvision.transforms.Compose:
    """
    Returns a torchvision.transforms.Compose object that transforms the data.
    """
    transform = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
    ])
    return transform

def test_image_foldering(data_dir: str, batch: int, size: int) -> torchvision.datasets.ImageFolder:
    """
    Returns a torchvision.datasets.ImageFolder object that contains the testing data.
    """

    test_set = torchvision.datasets.ImageFolder(
        root='data/Gender/testing', 
        transform=test_transforms_data(batch, size)
    )
    return test_set

def test_data_loader(test_set: torchvision.datasets.ImageFolder,) -> DataLoader:
    """
    Returns a torch.utils.data.DataLoader object that contains the data.
    """
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )
    return test_loader