from torch import no_grad, stack, float, norm, tensor, argmin
from torch.nn.functional import normalize
from sklearn.cluster import KMeans
from icarl.config import exemplars_nr_per_class, seed

def select_exemplars_with_kmeans(model, device, current_original_train_data, class_means_or_medians):
    new_memory_dataset = []
    exemplars_per_label = {}

    for features, target in current_original_train_data:
        if target not in exemplars_per_label:
            exemplars_per_label[target] = [features]
        else:
            exemplars_per_label[target].append(features)

    model.eval()

    with no_grad():
        for target, exemplars in exemplars_per_label.items():
            exemplars_tensor = stack(exemplars).to(device)
            features = model(exemplars_tensor, return_features=True)
            normalized_features = [normalize(feature, dim=0) for feature in features]
            features_np = stack(normalized_features).cpu().detach().numpy()
            # TODO: set random_state to seed from config
            kmeans = KMeans(n_clusters=exemplars_nr_per_class, random_state=42, n_init=50).fit(features_np)
            centers = kmeans.cluster_centers_
            centers_tensor = tensor(centers, dtype=float)
            selected_exemplars = []

            for center in centers_tensor:
                center = center.unsqueeze(0).repeat(len(features_np), 1)
                distances = norm(tensor(features_np) - center, dim=1)
                nearest_index = argmin(distances).item()
                selected_exemplars.append(exemplars[nearest_index])

            selected_features = model(stack(selected_exemplars).to(device), return_features=True)
            normalized_selected_features = [normalize(feature, dim=0) for feature in selected_features]
            class_means_or_medians[target] = normalize(stack(normalized_selected_features).mean(dim=0), dim=0)

            # Update memory dataset
            for features in selected_exemplars:
                new_memory_dataset.append((features, target))

    return new_memory_dataset