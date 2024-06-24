from torch import no_grad, stack, tensor, float, norm, argmin
from torch.nn.functional import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .config import exemplars_nr_per_target, seed, visualize_exemplar_selection
from .visualize_exemplar_selection_with_k_means import visualize_exemplar_selection_with_k_means


def select_exemplars_with_k_means(model, device, train_data_grouped_by_targets, target_means_or_medians):
    selected_exemplars = []

    model.eval()

    with no_grad():
        for target, target_samples in train_data_grouped_by_targets.items():
            features = stack(target_samples).to(device)
            returned_features = model(features, return_features=True).cpu().detach()
            returned_features_normalized = normalize(returned_features)

            if visualize_exemplar_selection:
                pca = PCA(n_components=2)
                returned_features_normalized = pca.fit_transform(returned_features_normalized)
                returned_features_normalized = tensor(returned_features_normalized)

            kmeans = KMeans(n_clusters=exemplars_nr_per_target, random_state=seed, n_init=50)

            kmeans.fit(returned_features_normalized)

            centers = kmeans.cluster_centers_
            centers_tensor = tensor(centers, dtype=float)
            selected_exemplar_features = []
            selected_exemplar_indices = []

            for center_tensor in centers_tensor:
                distances = norm(returned_features_normalized - center_tensor, dim=1)
                nearest_index = argmin(distances).item()

                selected_exemplar_indices.append(nearest_index)
                selected_exemplar_features.append(target_samples[nearest_index])

            for features in selected_exemplar_features:
                selected_exemplars.append([features, target])

            if visualize_exemplar_selection:
                visualize_exemplar_selection_with_k_means(returned_features_normalized, selected_exemplar_indices,
                                                          centers, kmeans.labels_, target)
            else:
                selected_features = model(stack(selected_exemplar_features).to(device), return_features=True)
                normalized_selected_features = normalize(selected_features, dim=1)
                target_means_or_medians[target] = normalize(normalized_selected_features.mean(dim=0), dim=0)

    if visualize_exemplar_selection:
        # If we had made visualizations where PCA dimensionality reduction was applied, since information is partially
        # lost in dimensionality reduction, we would not want to continue the algorithm. Plotting the selection of
        # exemplars serves to deepen our understanding of the process rather than completing the class-incremental
        # learning.
        exit()

    return selected_exemplars
