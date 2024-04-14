import torch
from torchvision.transforms import ToTensor
from torchvision import models, datasets
from torch.utils.data import TensorDataset, DataLoader
import random
import time as time
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans

TOTAL_TASKS_NR = 5
MAX_INSTANCES_PER_CLASS = 500
MEMORY_SIZE_PER_CLASS = 20
BATCH_SIZE = 128
EPOCHS = 70
LEARNING_RATE = 2.0
T = 2
SEED = 42

LOG_TO_FILE = True
RESULTS_FOLDER_NAME = 'results'
f = None

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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


def log(text, end='\n'):
    print(text, end=end)

    if LOG_TO_FILE:
        f.write(f"{text}{end}")


def get_current_date_time():
    current_time_seconds = time.time()
    current_time_datetime = datetime.fromtimestamp(current_time_seconds)

    return current_time_datetime.strftime('%Y_%m_%d_%H_%M_%S')


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


def select_new_exemplars_by_herding(device, model, dataset):
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
        features_np = torch.stack(normalized_features).cpu().detach().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit(features_np)
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


def select_exemplars_by_median(device, model, dataset, n_exemplars=MEMORY_SIZE_PER_CLASS):
    exemplars_means_m = {}
    new_memory_dataset = []

    model.eval()

    with torch.no_grad():
        for label in set(label for _, label in dataset):
            class_samples = [data for data, lbl in dataset if lbl == label]
            features = torch.stack(
                [model(data.unsqueeze(0).to(device), extract_features=True).squeeze() for data in class_samples]
            )

            # Compute the median of the class features
            median = features.median(dim=0).values
            exemplars_means_m[label] = median

            distances = torch.norm(features - median, dim=1)
            closest_indices = distances.topk(n_exemplars, largest=False).indices

            for idx in closest_indices:
                new_memory_dataset.append((class_samples[idx], label))

    return new_memory_dataset


def select_exemplars_by_density(device, model, dataset, n_exemplars=MEMORY_SIZE_PER_CLASS):
    new_memory_dataset = []
    exemplars_per_label = {}

    model.eval()

    with torch.no_grad():
        for data, label in dataset:
            if label not in exemplars_per_label:
                exemplars_per_label[label] = [data]
            else:
                exemplars_per_label[label].append(data)

        for label, exemplars in exemplars_per_label.items():
            exemplars_tensor = torch.stack(exemplars).to(device)
            features = model(exemplars_tensor, extract_features=True)

            # Compute the density of each exemplar
            density_scores = torch.sum(-torch.cdist(features, features, p=2), dim=1)

            # Select exemplars with the highest density
            selected_indices = torch.topk(density_scores, k=n_exemplars).indices
            selected_exemplars = [exemplars[i] for i in selected_indices]

            for exemplar in selected_exemplars:
                new_memory_dataset.append((exemplar, label))

    return new_memory_dataset


def create_tensor_dataset(dataset):
    data = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(torch.stack(data), labels_tensor)


def get_datasets(device, model, task_nr, exemplar_selection_method):
    global current_train_dataset
    global current_test_dataset
    global memory_dataset

    current_train_dataset = train_datasets_grouped_by_tasks[task_nr]
    current_test_dataset += test_datasets_grouped_by_tasks[task_nr]

    tensor_datasets = create_tensor_dataset(current_train_dataset + memory_dataset), create_tensor_dataset(
        current_test_dataset)

    memory_dataset += exemplar_selection_method(device, model, current_train_dataset)

    return tensor_datasets


def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def get_model(device):
    base_model = models.resnet18(weights=None)

    return ModelWrapper(device, base_model)


def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.00001)


def get_loss_fn():
    return torch.nn.CrossEntropyLoss()


def get_knowledge_distillation_loss(pred, soft):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)

    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def adjust_learning_rate(optimizer, epoch):
    if epoch == 49 or epoch == 63:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5


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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

    log(f"accuracy: {accuracy:.2f}, avg loss: {test_loss:.5f}", ' ')

    if epoch == EPOCHS - 1:
        accuracies_per_tasks.append(round(accuracy, 2))


def train(device, model, old_model, train_data_loader, test_data_loader, first_training):
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()

    log('')
    log(f"Total instances for training: {len(train_data_loader.dataset)}")
    log(f"Total instances for testing: {len(test_data_loader.dataset)}")
    log('')

    for epoch in range(EPOCHS):
        log("Epoch {:2d} - ".format(epoch + 1), '')

        start_time = time.time()

        adjust_learning_rate(optimizer, epoch)
        train_loop(train_data_loader, device, model, old_model, loss_fn, optimizer, first_training)
        test_loop(test_data_loader, device, model, loss_fn, epoch)

        end_time = time.time()
        epoch_duration = end_time - start_time

        log(f"({epoch_duration:.2f}s)")

    log('')


def incremental_learning(exemplar_selection_method):
    device = get_device()
    model = get_model(device)
    old_model = get_model(device)

    for task_nr in range(TOTAL_TASKS_NR):
        log(f"Task {task_nr + 1}:")

        train_data, test_data = get_datasets(device, model, task_nr, exemplar_selection_method)

        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        train(device, model, old_model, train_data_loader, test_data_loader, task_nr == 0)

        old_model.load_state_dict(model.state_dict())


def main(exemplar_selection_method):
    global f, TOTAL_TASKS_NR, MAX_INSTANCES_PER_CLASS, MEMORY_SIZE_PER_CLASS, BATCH_SIZE, EPOCHS, LEARNING_RATE, T, SEED

    if LOG_TO_FILE:
        f = open(f'{RESULTS_FOLDER_NAME}/{get_current_date_time()}-{exemplar_selection_method.__name__}.txt', 'x')

    log("Hyperparameters:")
    log('')
    log(f"Total tasks numbers: {TOTAL_TASKS_NR}")
    log(f"Max instances per class: {MAX_INSTANCES_PER_CLASS}")
    log(f"Memory size per class: {MEMORY_SIZE_PER_CLASS}")
    log(f"Batch size: {BATCH_SIZE}")
    log(f"Epochs: {EPOCHS}")
    log(f"Learning rate: {LEARNING_RATE}")
    log(f"T: {T}")
    log(f"Seed: {SEED}")
    log('')

    group_datasets_by_tasks(datasets.CIFAR100)
    incremental_learning(exemplar_selection_method)

    log(f"Accuracies per tasks: {accuracies_per_tasks}")

    if LOG_TO_FILE:
        f.close()


exemplar_selection_methods = {
    'HERDING': select_new_exemplars_by_herding,
    'KMEANS': select_exemplars_with_kmeans,
    'MEDIAN': select_exemplars_by_median,
    'DENSITY': select_exemplars_by_density,
}

if __name__ == '__main__':
    # main(exemplar_selection_methods['HERDING'])
    main(exemplar_selection_methods['KMEANS'])
    # main(exemplar_selection_methods['MEDIAN'])
    # main(exemplar_selection_methods['DENSITY'])
