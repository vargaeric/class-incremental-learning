import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from constants import ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
from .config import (
    epochs_nr, learning_rate_starting_value, learning_rate_milestones, batch_size, weight_decay, momentum
)
from .ResNet32 import ResNet32
from .train_one_epoch import train_one_epoch
from .define_class_means import define_class_means
from .model import get_accuracy_2


def group_training_data_by_classes(training_data_grouped_by_classes, current_original_train_data):
    # TODO: maybe add only the features
    for current_original_train_data in current_original_train_data:
        feature, target = current_original_train_data

        if target in training_data_grouped_by_classes:
            training_data_grouped_by_classes[target].append([feature, target])
        else:
            training_data_grouped_by_classes[target] = [[feature, target]]

def train_one_incremental_learning_step(device, model, old_model, loss_fn, task_nr, classes_nr, gamma,
                                        train_data_per_tasks, test_data_per_tasks, current_test_data, exemplars,
                                        training_data_grouped_by_classes, class_means, selection_method):
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
        train_one_epoch(model, old_model, device, train_data_loader, loss_fn, optimizer, scheduler, test_data_loader,
                  epoch_nr, task_nr, classes_nr)

    # TODO: prettify/optimize
    if task_nr == 0:
        old_model = ResNet32(classes_nr)
        old_model = old_model.to(device)

    old_model.load_state_dict(model.state_dict())

    group_training_data_by_classes(training_data_grouped_by_classes, current_original_train_data)

    new_exemplars = define_class_means(model, device, task_nr, current_original_train_data,
                                       training_data_grouped_by_classes, class_means, selection_method)

    print(f"New exemplars nr:{len(new_exemplars)}")

    ncm = get_accuracy_2(model, device, loss_fn, test_data_loader, 'feature_extractor', class_means)

    print('NCM: ', ncm)

    if selection_method == ORIGINAL_SELECTION or selection_method == MEDIAN_SELECTION:
        exemplars += new_exemplars
    else:
        exemplars = new_exemplars

    return old_model, exemplars
