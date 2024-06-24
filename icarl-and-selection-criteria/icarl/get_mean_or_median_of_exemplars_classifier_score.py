from torch import zeros, no_grad, cdist, argmax, mean, tensor, float
from torch.nn.functional import normalize
from .inference import inference


def get_mean_or_median_of_exemplars_classifier_score(model, device, test_data_loader, target_means_or_medians,
                                                     targets_nr, extracted_features_nr=64):
    target_mean_or_median_tensors = zeros(extracted_features_nr, targets_nr).to(device)

    for target, target_mean_or_median in target_means_or_medians.items():
        target_mean_or_median_tensors[:, target] = target_mean_or_median.clone().detach().to(device)

    target_mean_or_median_tensors_transposed = target_mean_or_median_tensors.T
    mean_or_median_of_exemplars_classifier_scores = []

    with no_grad():
        for features_in_batches, targets_in_batches in test_data_loader:
            features_in_batches = features_in_batches.to(device)
            returned_features_in_batches = inference(model, device, features_in_batches, True)
            returned_features_in_batches_transposed = returned_features_in_batches.T
            distances_between_target_means_or_medians_and_returned_features_in_batches = cdist(
                target_mean_or_median_tensors_transposed, normalize(returned_features_in_batches_transposed, dim=0).T
            )
            distances_between_target_means_or_medians_and_returned_features_in_batches = (
                -distances_between_target_means_or_medians_and_returned_features_in_batches
            ).T.detach().cpu()

            for batch_nr in range(len(distances_between_target_means_or_medians_and_returned_features_in_batches)):
                distances_between_target_means_or_medians_and_returned_features = \
                    distances_between_target_means_or_medians_and_returned_features_in_batches[batch_nr]
                targets = targets_in_batches[batch_nr]

                mean_or_median_of_exemplars_classifier_scores.append(
                    argmax(distances_between_target_means_or_medians_and_returned_features) == targets
                )

    return mean(tensor(mean_or_median_of_exemplars_classifier_scores, dtype=float)).detach().item()
