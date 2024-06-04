from torchvision.transforms import ToTensor
from .assign_data_to_tasks import assign_data_to_tasks


def group_data_by_tasks(data, tasks_nr, targets_order):
    data_folder_name = 'data'
    train_data = data(
        root=data_folder_name,
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = data(
        root=data_folder_name,
        train=False,
        download=True,
        transform=ToTensor()
    )

    print('Grouping training data by tasks...')

    train_data_grouped_by_tasks = assign_data_to_tasks(train_data, targets_order, tasks_nr)

    print('Grouping testing data by tasks...')

    test_data_grouped_by_tasks = assign_data_to_tasks(test_data, targets_order, tasks_nr)

    return train_data_grouped_by_tasks, test_data_grouped_by_tasks
