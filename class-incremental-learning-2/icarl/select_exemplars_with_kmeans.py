from torch import no_grad, stack, float, norm, tensor, argmin
from torch.nn.functional import normalize
from sklearn.cluster import KMeans
from icarl.config import exemplars_nr_per_target, seed

def select_exemplars_with_kmeans(model, device, train_data_grouped_by_targets, target_means_or_medians):
    selected_exemplars = []

    model.eval()

    with no_grad():
        for target, target_exemplars in train_data_grouped_by_targets.items():
            target_exemplars_tensor = stack(target_exemplars).to(device)
            features = model(target_exemplars_tensor, return_features=True)
            normalized_features = [normalize(feature, dim=0) for feature in features]
            normalized_features = stack(normalized_features).cpu().detach()
            # TODO: set random_state to seed from config
            kmeans = KMeans(n_clusters=exemplars_nr_per_target, random_state=42, n_init=50).fit(normalized_features)
            centers = kmeans.cluster_centers_
            centers_tensor = tensor(centers, dtype=float)
            selected_exemplars_okkkkk = []

            for center in centers_tensor:
                center = center.unsqueeze(0).repeat(len(normalized_features), 1)
                distances = norm(normalized_features  - center, dim=1)
                nearest_index = argmin(distances).item()

                selected_exemplars_okkkkk.append(target_exemplars[nearest_index])

            selected_features = model(stack(selected_exemplars_okkkkk).to(device), return_features=True)
            normalized_selected_features = [normalize(feature, dim=0) for feature in selected_features]
            target_means_or_medians[target] = normalize(stack(normalized_selected_features).mean(dim=0), dim=0)

            # Update memory dataset
            for features in selected_exemplars_okkkkk:
                selected_exemplars.append((features, target))

    return selected_exemplars
