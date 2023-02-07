from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def read_cifar10(batch_size, data_dir):
    # 数据变换
    trasnform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # 填充后裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]
    )

    # 数据集
    data_train = datasets.CIFAR10(root=data_dir, train=True, transform=trasnform_train, download=False)
    data_test = datasets.CIFAR10(root=data_dir, train=False, transform=transform_test, download=False)

    # 数据加载
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader

