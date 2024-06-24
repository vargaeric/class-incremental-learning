from torch import no_grad, stack, tensor, median, mean, sum, cdist, norm, topk
from torch.nn.functional import normalize
from sklearn.decomposition import PCA
from constants import MEDIAN_SELECTION, DENSITY_SELECTION
from .config import exemplars_nr_per_target, tasks_nr, targets_order, visualize_exemplar_selection
from .visualize_exemplar_selection_by_mean_or_median_or_density import visualize_exemplar_selection_by_mean_or_median_or_density


def select_exemplars_by_mean_or_median_or_density(model, device, task_nr, train_data_grouped_by_targets,
                                                  target_means_or_medians, selection_method):
    selected_exemplars = []
    targets_nr_in_one_task = int(len(targets_order) / tasks_nr)
    current_task_targets = targets_order[task_nr * targets_nr_in_one_task:(task_nr + 1) * targets_nr_in_one_task]

    model.eval()

    with no_grad():
        for current_task_target in current_task_targets:
            features = stack(train_data_grouped_by_targets[current_task_target]).to(device)
            returned_features = model(features, return_features=True).detach().cpu()
            returned_features_normalized = normalize(returned_features)

            if visualize_exemplar_selection:
                pca = PCA(n_components=2)
                returned_features_normalized = pca.fit_transform(returned_features_normalized)
                returned_features_normalized = tensor(returned_features_normalized)

            if selection_method == MEDIAN_SELECTION:
                mean_or_median = median(returned_features_normalized, dim=0)[0]
            else:
                mean_or_median = mean(returned_features_normalized, dim=0)

            target_means_or_medians[current_task_target] = mean_or_median

            if selection_method == DENSITY_SELECTION:
                distances = sum(-cdist(returned_features_normalized, returned_features_normalized), dim=1)
            else:
                distances = -norm(returned_features_normalized - mean_or_median, dim=1)

            most_similar_exemplar_indices = topk(distances, exemplars_nr_per_target).indices

            if visualize_exemplar_selection:
                selected_exemplars_based_on_pca = returned_features_normalized[most_similar_exemplar_indices.numpy()]

                visualize_exemplar_selection_by_mean_or_median_or_density(returned_features_normalized,
                                                                          selected_exemplars_based_on_pca,
                                                                          mean_or_median, current_task_target,
                                                                          selection_method)

            selected_exemplars += [[train_data_grouped_by_targets[current_task_target][index], current_task_target] for
                                   index in most_similar_exemplar_indices]

    if visualize_exemplar_selection:
        # If we had made visualizations where PCA dimensionality reduction was applied, since information is partially
        # lost in dimensionality reduction, we would not want to continue the algorithm. Plotting the selection of
        # exemplars serves to deepen our understanding of the process rather than completing the class-incremental
        # learning.
        exit()

    return selected_exemplars
