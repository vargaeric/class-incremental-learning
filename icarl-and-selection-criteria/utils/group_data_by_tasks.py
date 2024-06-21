from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100, Food101
from constants import MNIST_DATASET, FMNIST_DATASET, CIFAR100_DATASET, FOOD101_DATASET
from .assign_data_to_tasks import assign_data_to_tasks


def group_data_by_tasks(dataset, tasks_nr, targets_order):
    data_folder_name = 'data'
    transform = Compose([
        Resize((32, 32)),
        ToTensor(),
        lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
    ])

    if dataset == MNIST_DATASET or dataset == FMNIST_DATASET or dataset == CIFAR100_DATASET:
        if dataset == MNIST_DATASET:
            data = MNIST
        elif dataset == FMNIST_DATASET:
            data = FashionMNIST
        else:
            data = CIFAR100

        train_data = data(
            root=data_folder_name,
            train=True,
            download=True,
            transform=transform
        )
        test_data = data(
            root=data_folder_name,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == FOOD101_DATASET:
        train_data = Food101(
            root=data_folder_name,
            split='train',
            download=True,
            transform=transform
        )
        test_data = Food101(
            root=data_folder_name,
            split='test',
            download=True,
            transform=transform
        )
    else:
        raise NotImplementedError("Dataset not implemented")

    train_data_grouped_by_tasks = assign_data_to_tasks(train_data, targets_order, tasks_nr)
    test_data_grouped_by_tasks = assign_data_to_tasks(test_data, targets_order, tasks_nr)

    return train_data_grouped_by_tasks, test_data_grouped_by_tasks
