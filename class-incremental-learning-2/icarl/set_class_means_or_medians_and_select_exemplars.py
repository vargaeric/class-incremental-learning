from constants import ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
from .set_class_means_or_medians import set_class_means_or_medians
from .select_exemplars_with_kmeans import select_exemplars_with_kmeans


def set_class_means_or_medians_and_select_exemplars(model, device, task_nr, current_original_train_data,
                                                    training_data_grouped_by_classes, class_means_or_medians,
                                                    selection_method):
    selected_exemplars = []

    if (selection_method == ORIGINAL_SELECTION or selection_method == MEDIAN_SELECTION
            or selection_method == DENSITY_SELECTION):
        selected_exemplars += set_class_means_or_medians(model, device, task_nr, current_original_train_data,
                                                         training_data_grouped_by_classes, class_means_or_medians,
                                                         selection_method)
    else:
        selected_exemplars = select_exemplars_with_kmeans(model, device, current_original_train_data,
                                                          class_means_or_medians)

    return selected_exemplars
