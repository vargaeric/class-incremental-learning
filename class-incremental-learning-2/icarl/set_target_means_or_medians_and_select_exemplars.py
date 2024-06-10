from constants import ORIGINAL_SELECTION, MEDIAN_SELECTION, DENSITY_SELECTION
from .set_target_means_or_medians import set_target_means_or_medians
from .select_exemplars_with_k_means import select_exemplars_with_k_means


def set_target_means_or_medians_and_select_exemplars(model, device, task_nr, train_data_grouped_by_targets,
                                                     target_means_or_medians, selection_method):
    selected_exemplars = []

    if (selection_method == ORIGINAL_SELECTION or selection_method == MEDIAN_SELECTION
            or selection_method == DENSITY_SELECTION):
        selected_exemplars += set_target_means_or_medians(model, device, task_nr, train_data_grouped_by_targets,
                                                          target_means_or_medians, selection_method)
    else:
        selected_exemplars = select_exemplars_with_k_means(model, device, train_data_grouped_by_targets,
                                                          target_means_or_medians)

    return selected_exemplars
