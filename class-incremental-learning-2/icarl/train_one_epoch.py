from torch import float
from torch.nn.functional import one_hot
from .config import tasks_nr, classes_order
from .inference import inference
from .train import train
from .get_accuracy_and_loss import get_accuracy_and_loss


def train_one_epoch(model, old_model, device, train_data_loader, loss_fn, optimizer, scheduler, test_data_loader,
                    epoch_nr, task_nr, classes_nr):
    train_loss = 0

    print(f"Epoch {epoch_nr}: ", end='')

    for features_in_batches, targets_in_batches in train_data_loader:
        targets_in_batches = one_hot(targets_in_batches, classes_nr).to(dtype=float)
        features_in_batches = features_in_batches.to(device)
        targets_in_batches = targets_in_batches.to(device)

        if task_nr != 0:
            predicted_targets_in_batches = inference(old_model, device, features_in_batches)
            previous_classes = classes_order[:(task_nr * tasks_nr)]
            targets_in_batches[:, previous_classes] = predicted_targets_in_batches[:, previous_classes]

        train_loss += train(model, device, loss_fn, optimizer, features_in_batches, targets_in_batches)

    accuracy, loss = get_accuracy_and_loss(model, device, loss_fn, test_data_loader, classes_nr)

    print(f'train loss - {train_loss} | val loss - {loss} | accuracy - {accuracy}')

    scheduler.step()
