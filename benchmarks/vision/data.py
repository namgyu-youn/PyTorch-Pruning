from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader():
    """Get CIFAR-10 test dataset for evaluation"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)