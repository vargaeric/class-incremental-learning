import torch
import numpy as np
from sklearn.cluster import KMeans

def select_exemplars_with_kmeans(model, device, dataset, n_clusters, class_means):
    def l2_normalization(vector):
        return torch.nn.functional.normalize(vector, p=2, dim=0)

    new_memory_dataset = []
    exemplars_per_label = {}

    # Step 1: Extract features for the dataset
    model.eval()

    with torch.no_grad():
        for data, label in dataset:
            if label not in exemplars_per_label:
                exemplars_per_label[label] = [data]
            else:
                exemplars_per_label[label].append(data)

    # Step 2 and 3: Apply K-Means on normalized features
    for label, exemplars in exemplars_per_label.items():
        exemplars_tensor = torch.stack(exemplars).to(device)
        # features = model(exemplars_tensor, extract_features=True)

        features = model(exemplars_tensor, return_features=True)

        normalized_features = [l2_normalization(feature) for feature in features]
        features_np = torch.stack(normalized_features).cpu().detach().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features_np)
        centers = kmeans.cluster_centers_

        # Step 4: Select nearest exemplars to cluster centers
        selected_exemplars = []

        for center in centers:
            distances = np.linalg.norm(features_np - center, axis=1)
            nearest_index = np.argmin(distances)
            selected_exemplars.append(exemplars[nearest_index])

        selected_features = model(torch.stack(selected_exemplars).to(device), return_features=True)

        normalized_selected_features = [l2_normalization(feature) for feature in selected_features]
        class_means[label] = l2_normalization(torch.stack(normalized_selected_features).mean(dim=0))

        # Update memory dataset
        for data in selected_exemplars:
            new_memory_dataset.append((data, label))

    return new_memory_dataset
