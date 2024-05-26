from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from constants import ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
from .config import (
    epochs_nr, learning_rate_starting_value, learning_rate_milestones, batch_size, weight_decay, momentum
)
from .ResNet32 import ResNet32
from .train_one_epoch import train_one_epoch
from .group_training_data_by_classes import group_training_data_by_classes
from .set_class_means_or_medians_and_select_exemplars import set_class_means_or_medians_and_select_exemplars
from .get_mean_of_exemplars_classifier_score import get_mean_of_exemplars_classifier_score


def train_one_incremental_learning_step(model, old_model, device, loss_fn, task_nr, classes_nr, gamma,
                                        train_data_grouped_by_tasks, test_data_grouped_by_tasks, current_test_data,
                                        exemplars, train_data_grouped_by_classes, class_means_or_medians,
                                        selection_method):
    current_original_train_data = train_data_grouped_by_tasks[task_nr]
    current_train_data = train_data_grouped_by_tasks[task_nr]
    current_train_data += exemplars
    current_test_data += test_data_grouped_by_tasks[task_nr]

    print(f"Task {task_nr}:")
    print(f"Current train data length: {len(current_train_data)}")
    print(f"Current test data length: {len(current_test_data)}")

    train_data_loader = DataLoader(current_train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(current_test_data, batch_size=batch_size, shuffle=True)
    optimizer = SGD(model.parameters(), lr=learning_rate_starting_value, weight_decay=weight_decay, momentum=momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, learning_rate_milestones, gamma=gamma)

    model.train()

    for epoch_nr in range(epochs_nr):
        train_one_epoch(model, old_model, device, train_data_loader, loss_fn, optimizer, scheduler, test_data_loader,
                  epoch_nr, task_nr, classes_nr)

    if task_nr == 0:
        old_model = ResNet32(classes_nr)
        old_model = old_model.to(device)

    old_model_parameters = model.state_dict()

    old_model.load_state_dict(old_model_parameters)
    group_training_data_by_classes(train_data_grouped_by_classes, current_original_train_data)

    new_exemplars = set_class_means_or_medians_and_select_exemplars(model, device, task_nr, current_original_train_data,
                                                                    train_data_grouped_by_classes,
                                                                    class_means_or_medians, selection_method)
    mean_of_exemplars_classifier_score = get_mean_of_exemplars_classifier_score(model, device, test_data_loader,
                                                                                class_means_or_medians, classes_nr)

    print('Mean-of-exemplars classifier\'s accuracy: ', mean_of_exemplars_classifier_score)

    if selection_method == ORIGINAL_SELECTION or selection_method == MEDIAN_SELECTION:
        exemplars += new_exemplars
    else:
        exemplars = new_exemplars

    return old_model, exemplars
