from torch import mv, topk
from icarl.config import exemplars_nr_per_target


def select_exemplars_by_target_mean_or_median(returned_features_transposed_normalized, returned_features_median_or_mean,
                                              train_data_for_target, current_task_target):
    returned_features_normalized = returned_features_transposed_normalized.T
    returned_features_similarities = mv(returned_features_normalized, returned_features_median_or_mean)
    _, most_similar_exemplar_indices = topk(returned_features_similarities, exemplars_nr_per_target)

    return [[train_data_for_target[index], current_task_target] for index in most_similar_exemplar_indices]
