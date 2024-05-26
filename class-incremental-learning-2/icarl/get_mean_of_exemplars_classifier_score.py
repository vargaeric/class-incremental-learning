from torch import zeros, no_grad, norm, cdist, nn, argmax, mean, tensor, float
from torch.nn.functional import normalize
from .config import batch_size
from .inference import inference


def get_mean_of_exemplars_classifier_score(model, device, test_data_loader, class_means_or_medians, classes_nr,
                                           extracted_features_nr = 64):
    class_mean_tensors = zeros(extracted_features_nr, classes_nr).to(device)

    for target, class_mean_or_median in class_means_or_medians.items():
        class_mean_tensors[:, target] = class_mean_or_median.clone().detach().to(device)

    class_mean_tensors_transposed = class_mean_tensors.T
    mean_of_exemplars_classifier_scores = []

    with no_grad():
        for features_in_batches, targets_in_batches in test_data_loader:
            features_in_batches = features_in_batches.to(device)
            returned_features_in_batches = inference(model, device, features_in_batches, True)
            returned_features_in_batches_transposed = returned_features_in_batches.T
            distances_between_class_means_and_returned_features_in_batches = cdist(
                class_mean_tensors_transposed, normalize(returned_features_in_batches_transposed, dim=0).T
            )
            distances_between_class_means_and_returned_features_in_batches = (
                -distances_between_class_means_and_returned_features_in_batches
            ).T.detach().cpu()

            for batch_nr in range(len(distances_between_class_means_and_returned_features_in_batches)):
                distances_between_class_means_and_returned_features = \
                    distances_between_class_means_and_returned_features_in_batches[batch_nr]
                targets = targets_in_batches[batch_nr]

                mean_of_exemplars_classifier_scores.append(
                    argmax(distances_between_class_means_and_returned_features) == targets
                )

    return mean(tensor(mean_of_exemplars_classifier_scores, dtype=float))
