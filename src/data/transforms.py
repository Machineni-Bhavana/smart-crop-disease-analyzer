import torchvision.transforms as T
from config import Config

def get_train_transforms(image_size: int = Config.IMAGE_SIZE) -> T.Compose:
    """
    Data augmentation and preprocessing transformations for training.
    Deterministic inference transforms must NOT be mixed here.
    """
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])

def get_eval_transforms(image_size: int = Config.IMAGE_SIZE) -> T.Compose:
    """
    Deterministic evaluation and test transformations.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])
