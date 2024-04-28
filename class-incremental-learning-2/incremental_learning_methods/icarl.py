import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from sklearn.cluster import KMeans

from incremental_learning_methods.icarl_net import make_icarl_net, initialize_icarl_net
from incremental_learning_methods.model import make_batch_one_hot, train, get_accuracy, inference, get_feature_extraction_layer, get_accuracy_2


def group_training_data_by_classes(training_data_grouped_by_classes, current_original_train_data):
    # TODO: maybe add only the features
    for current_original_train_data in current_original_train_data:
        feature, target = current_original_train_data

        if target in training_data_grouped_by_classes:
            training_data_grouped_by_classes[target].append([feature, target])
        else:
            training_data_grouped_by_classes[target] = [[feature, target]]


def get_current_task_classes(classes_order, tasks_nr, task_nr, ):
    return classes_order[task_nr * tasks_nr:(task_nr + 1) * tasks_nr]


def l2_normalization(vector):
    return torch.nn.functional.normalize(vector, p=2, dim=0)


def select_exemplars_with_kmeans(model, device, dataset, n_clusters, class_means):
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
        # features = model(exemplars_tensor, extract_features=True)

        features = get_feature_extraction_layer(
            model,
            device,
            'feature_extractor',
            exemplars_tensor,
        )

        normalized_features = [l2_normalization(feature) for feature in features]
        features_np = torch.stack(normalized_features).cpu().detach().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features_np)
        centers = kmeans.cluster_centers_

        # Step 4: Select nearest exemplars to cluster centers
        selected_exemplars = []

        for center in centers:
            distances = np.linalg.norm(features_np - center, axis=1)
            nearest_index = np.argmin(distances)
            selected_exemplars.append(exemplars[nearest_index])

        # Normalization and mean calculation for the selected exemplars
        # selected_features = model(torch.stack(selected_exemplars).to(device), extract_features=True)
        selected_features = get_feature_extraction_layer(
            model,
            device,
            'feature_extractor',
            torch.stack(selected_exemplars).to(device),
        )

        normalized_selected_features = [l2_normalization(feature) for feature in selected_features]
        class_means[label] = l2_normalization(torch.stack(normalized_selected_features).mean(dim=0))

        # Update memory dataset
        for data in selected_exemplars:
            new_memory_dataset.append((data, label))

    return new_memory_dataset


def define_class_means(model, device, exemplars_nr_per_class, classes_order, tasks_nr, task_nr, current_original_train_data, training_data_grouped_by_classes, class_means, herding=True):
    # TODO: remove :2
    current_task_classes = get_current_task_classes(classes_order, tasks_nr, task_nr)

    selected_exemplars = []

    for current_task_class in current_task_classes:
        print(f"Selection exemplars for {current_task_class}...")

        original_data = training_data_grouped_by_classes[current_task_class]
        features = [item[0] for item in training_data_grouped_by_classes[current_task_class]]
        features_tensor = torch.stack(features)

        extracted_features_from_last_layer = get_feature_extraction_layer(
            model,
            device,
            'feature_extractor',
            features_tensor,
        )
        D = extracted_features_from_last_layer.T
        D = D / torch.norm(D, dim=0)

        mu = torch.mean(D, dim=1)

        # Storing or using the index
        class_means[current_task_class] = mu  # Store the mean

        if herding:
            D_transposed = D.T  # Correct the orientation
            similarity = torch.mv(D_transposed, mu)  # Corrected Matrix-vector multiplication

            top_k_values, top_k_indices = torch.topk(similarity, exemplars_nr_per_class)

            selected_exemplars_for_class = [original_data[idx] for idx in top_k_indices.tolist()]

            selected_exemplars += selected_exemplars_for_class

    if not herding:
        selected_exemplars += select_exemplars_with_kmeans(model, device, current_original_train_data, exemplars_nr_per_class, class_means)

    return selected_exemplars


def icarl(exemplars_nr_per_class, epochs_nr, learning_rate_starting_value, learning_rate_division_value,
          learning_rate_milestones, batch_size, weight_decay, momentum, tasks_nr, classes_order, seed,
          grouped_data_by_tasks):
    torch.manual_seed(seed)

    classes_nr = len(classes_order)
    model = make_icarl_net(num_classes=classes_nr)

    model.apply(initialize_icarl_net)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    old_model = None
    loss_fn = BCELoss()
    gamma = 1.0 / learning_rate_division_value

    train_data_per_tasks, test_data_per_tasks = grouped_data_by_tasks
    current_test_data = []
    exemplars = []
    training_data_grouped_by_classes = {}
    class_means = {}

    for task_nr in range(tasks_nr):
        print(f"Task {task_nr}:")

        current_original_train_data = train_data_per_tasks[task_nr]
        current_train_data = train_data_per_tasks[task_nr]

        if task_nr != 0:
            current_train_data += exemplars

        print(f"Current train data length: {len(current_train_data)}")

        current_test_data += test_data_per_tasks[task_nr]

        print(f"Current test data length: {len(current_test_data)}")

        train_data_loader = DataLoader(current_train_data, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(current_test_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate_starting_value,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
        scheduler = MultiStepLR(optimizer, learning_rate_milestones, gamma=gamma)

        # TODO: prettify/optimize

        model.train()

        for epoch_nr in range(epochs_nr):
            total_loss = 0

            print(f"Epoch {epoch_nr}: ", end='')

            for features_in_batches, targets_in_batches in train_data_loader:
                targets_in_batches = make_batch_one_hot(targets_in_batches, classes_nr)

                # TODO: Maybe we dont need this
                features_in_batches = features_in_batches.to(device)
                targets_in_batches = targets_in_batches.to(device)

                if task_nr != 0:
                    predicted_target = inference(old_model, device, features_in_batches)
                    previous_classes = classes_order[:(task_nr * tasks_nr)]
                    targets_in_batches[:, previous_classes] = predicted_target[:, previous_classes]

                total_loss += train(model, device, loss_fn, optimizer, features_in_batches, targets_in_batches)

            accuracy, loss = get_accuracy(model, device, loss_fn, test_data_loader)

            print(f'train loss - {total_loss} | val loss - {loss} | accuracy - {accuracy}')

            scheduler.step()

        # TODO: prettify/optimize
        if task_nr == 0:
            old_model = make_icarl_net(classes_nr)
            old_model = old_model.to(device)

        old_model.load_state_dict(model.state_dict())

        group_training_data_by_classes(training_data_grouped_by_classes, current_original_train_data)

        new_exemplars = define_class_means(model, device, exemplars_nr_per_class, classes_order, tasks_nr, task_nr, current_original_train_data, training_data_grouped_by_classes, class_means, False)

        print(f"New exemplars nr:{len(new_exemplars)}")

        ncm = get_accuracy_2(model, device, loss_fn, test_data_loader, 'feature_extractor', class_means)

        print('NCM: ', ncm)

        exemplars = new_exemplars

    print('End!')

