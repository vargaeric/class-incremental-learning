from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from constants import ORIGINAL_SELECTION, MEDIAN_SELECTION
from .config import (
    epochs_nr, learning_rate_starting_value, learning_rate_milestones, batch_size, weight_decay, momentum
)
from .ResNet32iCaRL import ResNet32iCaRL
from .train_one_epoch import train_one_epoch
from .group_training_data_by_targets import group_training_data_by_targets
from .set_target_means_or_medians_and_select_exemplars import set_target_means_or_medians_and_select_exemplars
from .get_mean_or_median_of_exemplars_classifier_score import get_mean_or_median_of_exemplars_classifier_score


def train_one_incremental_learning_step(model, old_model, device, loss_fn, task_nr, targets_nr, gamma,
                                        current_train_data, current_test_data, exemplars, target_means_or_medians,
                                        selection_method, accuracy_scores, log):
    log()
    log(f"Task {task_nr + 1}:")
    log(f"Current train data length: {len(current_train_data)}")
    log(f"Current test data length: {len(current_test_data)}")

    train_data_loader = DataLoader(current_train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(current_test_data, batch_size=batch_size, shuffle=True)
    optimizer = SGD(model.parameters(), lr=learning_rate_starting_value, weight_decay=weight_decay, momentum=momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, learning_rate_milestones, gamma=gamma)

    model.train()

    for epoch_nr in range(epochs_nr):
        train_one_epoch(model, old_model, device, train_data_loader, loss_fn, optimizer, scheduler, test_data_loader,
                        epoch_nr, task_nr, targets_nr, log)

    if task_nr == 0:
        old_model = ResNet32iCaRL(targets_nr)
        old_model = old_model.to(device)

    old_model_parameters = model.state_dict()

    old_model.load_state_dict(old_model_parameters)

    train_data_grouped_by_targets = {}

    group_training_data_by_targets(train_data_grouped_by_targets, current_train_data)

    new_exemplars = set_target_means_or_medians_and_select_exemplars(model, device, task_nr,
                                                                     train_data_grouped_by_targets,
                                                                     target_means_or_medians, selection_method)
    mean_or_median_of_exemplars_classifier_score = get_mean_or_median_of_exemplars_classifier_score(
        model, device, test_data_loader, target_means_or_medians, targets_nr
    )
    mean_or_median_of_exemplars_classifier_score_rounded = round(mean_or_median_of_exemplars_classifier_score, 5)
    classifier_type = 'Median' if selection_method == MEDIAN_SELECTION else 'Mean'

    log(f"{classifier_type}-of-exemplars classifier's accuracy: {mean_or_median_of_exemplars_classifier_score_rounded}")
    accuracy_scores.append(mean_or_median_of_exemplars_classifier_score_rounded)

    if selection_method == ORIGINAL_SELECTION or selection_method == MEDIAN_SELECTION:
        exemplars += new_exemplars
    else:
        exemplars = new_exemplars

    return old_model, exemplars
