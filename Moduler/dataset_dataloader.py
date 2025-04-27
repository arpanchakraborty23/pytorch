"""
Contains funtality for crating pytorch dataset and dataloader for
Image Classification
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()

def create_dataloader(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = num_workers
):
    """
    Creating traing and testing dataloader

    Take inputs traning dir,  test dir  turn them into pytorch dataloader class

    Args: 
            train_dir:  Traning data path,
            test_dir: Test data path,
            transform: Perform transformation on data for traning (resize, to tensor, Agumentation),
            batch_size: num of sample in each traning batch,
            num_workers: num of cpu core working
    
    Return: Tranin dataloader , test dataloader , num classes in dataset
    """
    # creating pytorch dataset
    train_data = datasets.ImageFolder(train_dir,transform=transform)
    test_data = datasets.ImageFolder(test_dir,transform=transform)

    # get num of classes
    num_classes = train_data.classes
    print(f"Num of classes in dataset\n:{num_classes}")

    # Turn images into data loaders
    train_data_loader= DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_data_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_data_loader, test_data_loader, num_classes