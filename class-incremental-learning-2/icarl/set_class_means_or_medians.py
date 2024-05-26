from torch import stack, mean, median, mv, topk
from torch.nn.functional import normalize
from constants import MEDIAN_SELECTION, DENSITY_SELECTION
from icarl.config import exemplars_nr_per_class, tasks_nr, classes_order
from .select_exemplars_by_density import select_exemplars_by_density
from .select_exemplars_by_class_mean_or_median import select_exemplars_by_class_mean_or_median


def set_class_means_or_medians(model, device, task_nr, current_original_train_data, training_data_grouped_by_classes,
                               class_means_or_medians, selection_method):
    selected_exemplars = []
    classes_nr_in_one_task = int(len(classes_order) / tasks_nr)
    current_task_classes = classes_order[task_nr * classes_nr_in_one_task:(task_nr + 1) * classes_nr_in_one_task]

    for current_task_class in current_task_classes:
        training_data_for_class = training_data_grouped_by_classes[current_task_class]
        features = [item[0] for item in training_data_grouped_by_classes[current_task_class]]
        features_tensor = stack(features)
        returned_features_transposed = model(features_tensor.to(device), return_features=True).detach().cpu().T
        returned_features_transposed_normalized = normalize(returned_features_transposed, dim=0)

        if selection_method == MEDIAN_SELECTION:
            returned_features_median_or_mean = median(returned_features_transposed_normalized, dim=1)[0]
        else:
            returned_features_median_or_mean = mean(returned_features_transposed_normalized, dim=1)

        class_means_or_medians[current_task_class] = returned_features_median_or_mean

        if selection_method == DENSITY_SELECTION:
            selected_exemplars = select_exemplars_by_density(model, device, current_original_train_data,
                                                             exemplars_nr_per_class)
        else:
            selected_exemplars += select_exemplars_by_class_mean_or_median(returned_features_transposed_normalized,
                                                                           returned_features_median_or_mean,
                                                                           training_data_for_class)

    return selected_exemplars
