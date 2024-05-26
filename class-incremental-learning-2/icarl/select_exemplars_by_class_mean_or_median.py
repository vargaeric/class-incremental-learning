from torch import mv, topk
from icarl.config import exemplars_nr_per_class


def select_exemplars_by_class_mean_or_median(returned_features_transposed_normalized, returned_features_median_or_mean,
                                             training_data_for_class):
    returned_features_normalized = returned_features_transposed_normalized.T
    returned_features_similarities = mv(returned_features_normalized, returned_features_median_or_mean)
    _, most_similar_exemplar_indices = topk(returned_features_similarities, exemplars_nr_per_class)

    return [training_data_for_class[index] for index in most_similar_exemplar_indices]
