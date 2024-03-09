import torch
from torchvision.transforms import ToTensor
from torchvision import models, datasets
from torch.utils.data import TensorDataset, DataLoader
import random
import time as time

TOTAL_TASKS_NR = 5
MAX_INSTANCES_PER_CLASS = 100
MEMORY_SIZE_PER_CLASS = 1

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.01

classes_per_task = []
train_datasets_grouped_by_tasks = []
test_datasets_grouped_by_tasks = []
current_train_dataset = []
current_test_dataset = []
memory_dataset = []
accuracies_per_tasks = []


def split_list_equally(lst, num_parts):
    avg = len(lst) / float(num_parts)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out


def get_classes_per_task(dataset):
    classes_nr = len(dataset.classes)
    classes_representations = list(range(0, classes_nr))

    random.shuffle(classes_representations)

    return split_list_equally(classes_representations, TOTAL_TASKS_NR)


def get_class_task_relations():
    class_task_relations = {}

    for i, sublist in enumerate(classes_per_task):
        for class_repr in sublist:
            class_task_relations[class_repr] = i

    return class_task_relations


def get_datasets_grouped_by_tasks(dataset, class_task_relations, is_train_data=True):
    datasets_grouped_by_tasks = [[] for _ in range(TOTAL_TASKS_NR)]
    class_counts = {}

    for data, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0

        if (class_counts[label] < MAX_INSTANCES_PER_CLASS) or not is_train_data:
            datasets_grouped_by_tasks[class_task_relations[label]].append((data, label))
            class_counts[label] += 1

    return datasets_grouped_by_tasks


def group_datasets_by_tasks(dataset):
    global classes_per_task
    global train_datasets_grouped_by_tasks
    global test_datasets_grouped_by_tasks

    train_data = dataset(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    classes_per_task = get_classes_per_task(train_data)
    class_task_relations = get_class_task_relations()
    train_datasets_grouped_by_tasks = get_datasets_grouped_by_tasks(train_data, class_task_relations)

    test_data = dataset(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    test_datasets_grouped_by_tasks = get_datasets_grouped_by_tasks(test_data, class_task_relations, False)


def select_new_exemplars_for_memory_dataset(dataset):
    new_memory_dataset = []
    exemplars_per_class_counts = {}

    for data, label in dataset:
        if label not in exemplars_per_class_counts:
            exemplars_per_class_counts[label] = 0

        if exemplars_per_class_counts[label] < MEMORY_SIZE_PER_CLASS:
            new_memory_dataset.append((data, label))
            exemplars_per_class_counts[label] += 1

    return new_memory_dataset


def create_tensor_dataset(dataset):
    data = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(torch.stack(data), labels_tensor)


def get_datasets(task_nr):
    global current_train_dataset
    global current_test_dataset
    global memory_dataset

    memory_dataset += select_new_exemplars_for_memory_dataset(current_train_dataset)
    current_train_dataset = train_datasets_grouped_by_tasks[task_nr]
    current_test_dataset += test_datasets_grouped_by_tasks[task_nr]

    return create_tensor_dataset(current_train_dataset + memory_dataset), create_tensor_dataset(current_test_dataset)


def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def get_model(device):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 100)
    model = model.to(device)

    return model


def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def get_loss_fn():
    return torch.nn.CrossEntropyLoss()


def train_loop(train_data_loader, device, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(train_data_loader):
        X, y = X.to(device), y.to(device)

        # Make prediction and calculate the loss function
        pred = model(X)
        loss = loss_fn(pred, y)

        # Apply backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(test_data_loader, device, model, loss_fn, epoch):
    size = len(test_data_loader.dataset)
    batches_nr = len(test_data_loader)
    test_loss, correct = 0, 0

    model.eval()

    with torch.no_grad():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.max(dim=1).indices == y).type(torch.float).sum().item()

    test_loss /= batches_nr
    correct /= size
    accuracy = 100 * correct

    print(f"accuracy: {accuracy:.2f}, avg loss: {test_loss:.5f}", end=' ')

    if epoch == EPOCHS - 1:
        accuracies_per_tasks.append(round(accuracy, 2))


def train(train_data_loader, test_data_loader):
    device = get_device()
    model = get_model(device)
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()

    print()
    print(f"Total instances for training: {len(train_data_loader.dataset)}")
    print(f"Total instances for testing: {len(test_data_loader.dataset)}")
    print()

    for epoch in range(EPOCHS):
        print("Epoch {:2d} - ".format(epoch + 1), end='')

        start_time = time.time()

        train_loop(train_data_loader, device, model, loss_fn, optimizer)
        test_loop(test_data_loader, device, model, loss_fn, epoch)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"({epoch_duration:.2f}s)")

    print()


def incremental_learning():
    for task_nr in range(TOTAL_TASKS_NR):
        print(f"Task {task_nr + 1}:")

        train_data, test_data = get_datasets(task_nr)
        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        train(train_data_loader, test_data_loader)


def main():
    group_datasets_by_tasks(datasets.CIFAR100)
    incremental_learning()

    print('Accuracies per tasks: ', accuracies_per_tasks)


if __name__ == '__main__':
    main()
