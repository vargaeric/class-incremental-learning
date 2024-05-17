import torch

def select_exemplars_by_density(model, device, dataset, n_exemplars):
    new_memory_dataset = []
    exemplars_per_label = {}

    model.eval()

    with torch.no_grad():
        for data, label in dataset:
            if label not in exemplars_per_label:
                exemplars_per_label[label] = [data]
            else:
                exemplars_per_label[label].append(data)

        for label, exemplars in exemplars_per_label.items():
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