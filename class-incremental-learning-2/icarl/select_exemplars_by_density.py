import torch

def select_exemplars_by_density(model, device, train_data_grouped_by_targets, n_exemplars):
    new_memory_dataset = []

    model.eval()

    with torch.no_grad():
        for label, exemplars in train_data_grouped_by_targets.items():
            exemplars_tensor = torch.stack(exemplars).to(device)
            features = model(exemplars_tensor, return_features=True)

            # Compute the density of each exemplar
            density_scores = torch.sum(-torch.cdist(features, features, p=2), dim=1)

            # Select exemplars with the highest density
            selected_indices = torch.topk(density_scores, k=n_exemplars).indices
            selected_exemplars = [exemplars[i] for i in selected_indices]

            for exemplar in selected_exemplars:
                new_memory_dataset.append((exemplar, label))

    return new_memory_dataset