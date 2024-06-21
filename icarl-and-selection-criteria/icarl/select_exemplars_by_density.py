from torch import no_grad, stack, cdist, sum, topk
from icarl.config import exemplars_nr_per_target


def select_exemplars_by_density(model, device, train_data_grouped_by_targets):
    selected_exemplars = []

    model.eval()

    with no_grad():
        for target, target_exemplars in train_data_grouped_by_targets.items():
            target_exemplars_tensor = stack(target_exemplars).to(device)
            features = model(target_exemplars_tensor, return_features=True)
            density_scores = sum(-cdist(features, features, p=2), dim=1)
            selected_indices = topk(density_scores, k=exemplars_nr_per_target).indices

            selected_exemplars.extend((target_exemplars[index], target) for index in selected_indices)

    return selected_exemplars
