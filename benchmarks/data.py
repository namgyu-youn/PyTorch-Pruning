import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader(batch_size=32, subset_size=2000):
    """Get CIFAR-10 test dataset for evaluation"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # Use subset for faster evaluation
    if subset_size and subset_size < len(test_set):
        indices = torch.randperm(len(test_set))[:subset_size]
        test_set = torch.utils.data.Subset(test_set, indices)

    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)