from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def cifar10_loader(train=True, num_workers=1):
    # Prep Benchmark CIFAR-10 dataset to feed into ResNet 18
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to match ResNet-18's expected input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean and std
    ])

    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    if train:
        t_size = int(0.8 * len(dataset))
        v_size = len(dataset) - t_size
        train_dataset, val_dataset = random_split(dataset, [t_size, v_size])
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=num_workers, persistent_workers=True)
        return train_loader, val_loader
    else:
        loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers, persistent_workers=True)
        return loader