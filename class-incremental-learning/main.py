import torch
from torchvision.transforms import ToTensor
from torchvision import models, datasets
from torch.utils.data import TensorDataset, DataLoader
import random
import time as time
import numpy as np
from sklearn.cluster import KMeans

TOTAL_TASKS_NR = 5
MAX_INSTANCES_PER_CLASS = 100
MEMORY_SIZE_PER_CLASS = 1

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.01
T = 2
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Force PyTorch to use deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

classes_per_task = []
train_datasets_grouped_by_tasks = []
test_datasets_grouped_by_tasks = []
current_train_dataset = []
current_test_dataset = []
memory_dataset = []
accuracies_per_tasks = []
exemplars_means = {}


class ModelWrapper(torch.nn.Module):
    def __init__(self, device, model):
        super(ModelWrapper, self).__init__()

        self.features = torch.nn.Sequential(*list(model.children())[:-1])

        num_features = model.fc.in_features

        self.fc = torch.nn.Linear(num_features, 100)
        self.model = model

        self.model.to(device)
        self.fc.to(device)

    def forward(self, x, extract_features=False):
        x = self.features(x)
        x = torch.flatten(x, 1)

        if extract_features:
            return x
        else:
            return self.fc(x)


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


def l2_normalization(vector):
    return torch.nn.functional.normalize(vector, p=2, dim=0)


def euclidean_distance(first_vector, second_vector):
    return torch.norm(first_vector - second_vector, p=2)


def select_new_exemplars_for_memory_dataset(device, model, dataset):
    global exemplars_means

    new_memory_dataset = []
    exemplars_per_label = {}

    for data, label in dataset:
        if label not in exemplars_per_label:
            exemplars_per_label[label] = [data]
        else:
            exemplars_per_label[label].append(data)

    model.eval()

    with torch.no_grad():
        for index, (label, exemplar) in enumerate(exemplars_per_label.items()):
            results = model(torch.stack(exemplars_per_label[label]).to(device), extract_features=True)

            features_per_label = [l2_normalization(result) for result in results]

            # features_per_label = [l2_normalization(model(data.to(device).unsqueeze(0), extract_features=True)[0])
            #                       for data in exemplars_per_label[label]]

            exemplars_means[label] = l2_normalization(torch.stack(features_per_label).mean(dim=0))

            distances = [euclidean_distance(exemplars_means[label], feature) for feature in features_per_label]

            closest_features_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:MEMORY_SIZE_PER_CLASS]

            selected_exemplars = [exemplars_per_label[label][i] for i in closest_features_indices]

            for data in selected_exemplars:
                new_memory_dataset.append((data, label))

    return new_memory_dataset


def select_exemplars_with_kmeans(device, model, dataset, n_clusters=MEMORY_SIZE_PER_CLASS):
    global exemplars_means

    new_memory_dataset = []
    exemplars_per_label = {}

    # Step 1: Extract features for the dataset
    model.eval()

    with torch.no_grad():
        for data, label in dataset:
            if label not in exemplars_per_label:
                exemplars_per_label[label] = [data]
            else:
                exemplars_per_label[label].append(data)

    # Step 2 and 3: Apply K-Means on normalized features
    for label, exemplars in exemplars_per_label.items():
        exemplars_tensor = torch.stack(exemplars).to(device)
        features = model(exemplars_tensor, extract_features=True)
        normalized_features = [l2_normalization(feature) for feature in features]
        features_np = torch.stack(normalized_features).cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(features_np)
        centers = kmeans.cluster_centers_

        # Step 4: Select nearest exemplars to cluster centers
        selected_exemplars = []

        for center in centers:
            distances = np.linalg.norm(features_np - center, axis=1)
            nearest_index = np.argmin(distances)
            selected_exemplars.append(exemplars[nearest_index])

        # Normalization and mean calculation for the selected exemplars
        selected_features = model(torch.stack(selected_exemplars).to(device), extract_features=True)
        normalized_selected_features = [l2_normalization(feature) for feature in selected_features]
        exemplars_means[label] = l2_normalization(torch.stack(normalized_selected_features).mean(dim=0))

        # Update memory dataset
        for data in selected_exemplars:
            new_memory_dataset.append((data, label))

    return new_memory_dataset


def create_tensor_dataset(dataset):
    data = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(torch.stack(data), labels_tensor)


def get_datasets(device, model, task_nr):
    global current_train_dataset
    global current_test_dataset
    global memory_dataset

    current_train_dataset = train_datasets_grouped_by_tasks[task_nr]
    current_test_dataset += test_datasets_grouped_by_tasks[task_nr]

    tensor_datasets = create_tensor_dataset(current_train_dataset + memory_dataset), create_tensor_dataset(current_test_dataset)

    memory_dataset += select_new_exemplars_for_memory_dataset(device, model, current_train_dataset)

    return tensor_datasets


def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def get_model(device):
    base_model = models.resnet18(weights=None)

    return ModelWrapper(device, base_model)


def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def get_loss_fn():
    return torch.nn.CrossEntropyLoss()


def get_knowledge_distillation_loss(pred, soft):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)

    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def train_loop(train_data_loader, device, model, old_model, loss_fn, optimizer, first_training):
    model.train()

    for batch, (X, y) in enumerate(train_data_loader):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        if not first_training:
            old_model_logits = old_model(X)
            old_classes_nr = int(len(memory_dataset) / MEMORY_SIZE_PER_CLASS)

            loss += get_knowledge_distillation_loss(logits[:, :old_classes_nr], old_model_logits[:, :old_classes_nr])

        # print('loss: ', loss.item())

        # # Apply backpropagation
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        # Apply backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def nearest_mean_of_exemplars_classifier(feature):
    min_distance = float('inf')
    closest_key = None

    for label, label_features_mean in exemplars_means.items():
        distance = euclidean_distance(label_features_mean, feature)

        if distance < min_distance:
            min_distance = distance
            closest_key = label

    return closest_key


def test_loop(test_data_loader, device, model, loss_fn, epoch):
    size = len(test_data_loader.dataset)
    batches_nr = len(test_data_loader)
    test_loss, correct = 0, 0

    model.eval()

    with torch.no_grad():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            # features = model(X, extract_features=True)
            #
            # for i, feature in enumerate(features):
            #     pred = nearest_mean_of_exemplars_classifier(feature)
            #     correct += pred == y[i].item()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.max(dim=1).indices == y).type(torch.float).sum().item()

    test_loss /= batches_nr
    correct /= size
    accuracy = 100 * correct

    print(f"accuracy: {accuracy:.2f}, avg loss: {test_loss:.5f}", end=' ')

    if epoch == EPOCHS - 1:
        accuracies_per_tasks.append(round(accuracy, 2))


def train(device, model, old_model, train_data_loader, test_data_loader, first_training):
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()

    print()
    print(f"Total instances for training: {len(train_data_loader.dataset)}")
    print(f"Total instances for testing: {len(test_data_loader.dataset)}")
    print()

    for epoch in range(EPOCHS):
        print("Epoch {:2d} - ".format(epoch + 1), end='')

        start_time = time.time()

        train_loop(train_data_loader, device, model, old_model, loss_fn, optimizer, first_training)
        test_loop(test_data_loader, device, model, loss_fn, epoch)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"({epoch_duration:.2f}s)")

    print()


def incremental_learning():
    device = get_device()
    model = get_model(device)
    old_model = get_model(device)

    for task_nr in range(TOTAL_TASKS_NR):
        print(f"Task {task_nr + 1}:")

        train_data, test_data = get_datasets(device, model, task_nr)

        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        train(device, model, old_model, train_data_loader, test_data_loader, task_nr == 0)

        old_model.load_state_dict(model.state_dict())


def main():
    group_datasets_by_tasks(datasets.CIFAR100)
    incremental_learning()

    print('Accuracies per tasks: ', accuracies_per_tasks)


if __name__ == '__main__':
    main()
