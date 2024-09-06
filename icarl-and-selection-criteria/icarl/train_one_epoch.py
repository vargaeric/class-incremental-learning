from time import time
from torch import float
from torch.nn.functional import one_hot
from .config import tasks_nr, targets_order
from .inference import inference
from .train import train
from .get_accuracy_and_val_loss import get_accuracy_and_val_loss


def train_one_epoch(model, old_model, device, train_data_loader, loss_fn, optimizer, scheduler, test_data_loader,
                    epoch_nr, task_nr, targets_nr, log):
    train_loss = 0

    log(f"Epoch {epoch_nr + 1}: ", end='')

    epoch_start_time = time()

    for features_in_batches, targets_in_batches in train_data_loader:
        targets_in_batches = one_hot(targets_in_batches, targets_nr).to(dtype=float)
        features_in_batches = features_in_batches.to(device)
        targets_in_batches = targets_in_batches.to(device)

        if task_nr != 0:
            predicted_targets_in_batches = inference(old_model, device, features_in_batches)
            previous_targets = targets_order[:(task_nr * tasks_nr)]
            targets_in_batches[:, previous_targets] = predicted_targets_in_batches[:, previous_targets]

        train_loss += train(model, device, loss_fn, optimizer, features_in_batches, targets_in_batches)

    epoch_end_time = time()
    epoch_execution_time = epoch_end_time - epoch_start_time
    accuracy, val_loss = get_accuracy_and_val_loss(model, device, loss_fn, test_data_loader, targets_nr)

    log(f'training loss - {round(train_loss, 5)} | validation loss - {round(val_loss, 5)} | accuracy - '
        f'{round(accuracy, 5)} ({epoch_execution_time:.4f}s)')

    scheduler.step()
