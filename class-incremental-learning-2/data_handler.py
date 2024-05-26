from torchvision.transforms import ToTensor


def assign_data_to_tasks(data, classes_order, tasks_nr):
    data_grouped_by_tasks = [[] for _ in range(tasks_nr)]

    classes_nr = len(classes_order)
    classes_per_task = classes_nr // tasks_nr

    for (feature, target) in data:
        target_index_in_class_order = classes_order.index(target)

        task_index = target_index_in_class_order // classes_per_task

        if task_index >= tasks_nr:
            data_grouped_by_tasks[-1].append([feature, target])
        else:
            data_grouped_by_tasks[task_index].append([feature, target])

    return data_grouped_by_tasks


def group_data_by_tasks(data, tasks_nr, classes_order):
    DOWNLOAD_FOLDER_FOR_DATA = 'data'

    train_data = data(
        root=DOWNLOAD_FOLDER_FOR_DATA,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = data(
        root=DOWNLOAD_FOLDER_FOR_DATA,
        train=False,
        download=True,
        transform=ToTensor()
    )

    print('Grouping training data by tasks...')

    train_data_grouped_by_tasks = assign_data_to_tasks(train_data, classes_order, tasks_nr)

    print('Grouping testing data by tasks...')

    test_data_grouped_by_tasks = assign_data_to_tasks(test_data, classes_order, tasks_nr)

    return train_data_grouped_by_tasks, test_data_grouped_by_tasks
