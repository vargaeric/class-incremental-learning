from torch import stack, mean, median, mv, topk
from torch.nn.functional import normalize
from constants import MEDIAN_SELECTION, DENSITY_SELECTION
from icarl.config import exemplars_nr_per_target, tasks_nr, targets_order
from .select_exemplars_by_density import select_exemplars_by_density
from .select_exemplars_by_target_mean_or_median import select_exemplars_by_target_mean_or_median


def set_target_means_or_medians(model, device, task_nr, train_data_grouped_by_targets, target_means_or_medians,
                                selection_method):
    selected_exemplars = []
    targets_nr_in_one_task = int(len(targets_order) / tasks_nr)
    current_task_targets = targets_order[task_nr * targets_nr_in_one_task:(task_nr + 1) * targets_nr_in_one_task]

    for current_task_target in current_task_targets:
        features_tensor = stack(train_data_grouped_by_targets[current_task_target])
        returned_features_transposed = model(features_tensor.to(device), return_features=True).detach().cpu().T
        returned_features_transposed_normalized = normalize(returned_features_transposed, dim=0)

        if selection_method == MEDIAN_SELECTION:
            returned_features_median_or_mean = median(returned_features_transposed_normalized, dim=1)[0]
        else:
            returned_features_median_or_mean = mean(returned_features_transposed_normalized, dim=1)

        target_means_or_medians[current_task_target] = returned_features_median_or_mean

        if selection_method == DENSITY_SELECTION:
            selected_exemplars = select_exemplars_by_density(model, device, train_data_grouped_by_targets,
                                                             exemplars_nr_per_target)
        else:
            selected_exemplars += select_exemplars_by_target_mean_or_median(
                returned_features_transposed_normalized, returned_features_median_or_mean,
                train_data_grouped_by_targets[current_task_target], current_task_target
            )

    return selected_exemplars
