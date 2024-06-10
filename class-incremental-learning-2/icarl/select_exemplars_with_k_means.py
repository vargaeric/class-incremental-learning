from torch import no_grad, stack, float, norm, tensor, argmin
from torch.nn.functional import normalize
from sklearn.cluster import KMeans
from icarl.config import exemplars_nr_per_target, seed


def select_exemplars_with_k_means(model, device, train_data_grouped_by_targets, target_means_or_medians):
    selected_exemplars = []

    model.eval()

    with no_grad():
        for target, target_exemplars in train_data_grouped_by_targets.items():
            target_exemplars_tensor = stack(target_exemplars).to(device)
            features = model(target_exemplars_tensor, return_features=True)
            normalized_features = normalize(features, dim=1).cpu().detach()
            kmeans = KMeans(n_clusters=exemplars_nr_per_target, random_state=seed, n_init=50).fit(normalized_features)
            centers = kmeans.cluster_centers_
            centers_tensor = tensor(centers, dtype=float)
            selected_exemplar_features = []

            for center_tensor in centers_tensor:
                distances = norm(normalized_features - center_tensor, dim=1)
                nearest_index = argmin(distances).item()

                selected_exemplar_features.append(target_exemplars[nearest_index])

            for features in selected_exemplar_features:
                selected_exemplars.append((features, target))

            selected_features = model(stack(selected_exemplar_features).to(device), return_features=True)
            normalized_selected_features = normalize(selected_features, dim=1)
            target_means_or_medians[target] = normalize(normalized_selected_features.mean(dim=0), dim=0)

    return selected_exemplars
