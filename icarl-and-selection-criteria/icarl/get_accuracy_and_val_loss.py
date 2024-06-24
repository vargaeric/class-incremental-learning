from torch import no_grad, float, mean, tensor
from torch.nn.functional import one_hot


def get_accuracy_and_val_loss(model, device, loss_fn, test_data_loader, targets_nr):
    model.eval()

    scores = []
    val_loss = 0

    with no_grad():
        for features_in_batches, targets_in_batches in test_data_loader:
            model.zero_grad()

            features_in_batches = features_in_batches.to(device)
            targets_in_batches = targets_in_batches.to(device)
            predicted_targets_in_batches = model(features_in_batches)
            predicted_target = predicted_targets_in_batches.max(dim=1).indices
            scores += (predicted_target == targets_in_batches)
            targets_in_batches = one_hot(targets_in_batches, targets_nr).to(dtype=float)
            val_loss += loss_fn(predicted_targets_in_batches, targets_in_batches).detach().cpu().item()

    return mean(tensor(scores, dtype=float)).item(), val_loss
