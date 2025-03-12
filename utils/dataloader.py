from torch.utils.data import Dataset
from config.configs import *
import random
from torchvision import datasets
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class MyCustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        # Assuming x is a PIL Image, apply the transform here
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

def data_preparation_function(batch_size, path = train_path, train_ratio= train_size):
    # Set the random seed
    torch.manual_seed(42)
    random.seed(42)

    # Load the dataset without applying transform here
    full_dataset = datasets.ImageFolder(root=path)

    # Calculate sizes for train and test sets
    train_sizee = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_sizee

    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_sizee, val_size],
                                               generator=torch.Generator().manual_seed(42))

    # Wrap them in MyCustomDataset to apply the transform
    train_dataset = MyCustomDataset(train_dataset, transform=train_transforms)
    val_dataset = MyCustomDataset(val_dataset, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def test_loader_function(batch_size, test_path = None, transform = val_test_transform):
    # Create the ImageFolder dataset
    test_dataset = ImageFolder(root=test_path, transform=transform)

    # Create the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract the class_to_idx mapping from the dataset
    class_to_idx = test_dataset.class_to_idx
    # print(f"Found {len(test_dataset)} test images across {len(class_to_idx)} classes.")

    return test_loader, class_to_idx