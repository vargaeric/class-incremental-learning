from .config import tasks_nr, classes_order
from .model import make_batch_one_hot, train, get_accuracy, inference

def train_one_epoch(model, old_model, device, train_data_loader, loss_fn, optimizer, scheduler, test_data_loader,
              epoch_nr, task_nr, classes_nr):
    total_loss = 0

    print(f"Epoch {epoch_nr}: ", end='')

    for features_in_batches, targets_in_batches in train_data_loader:
        targets_in_batches = make_batch_one_hot(targets_in_batches, classes_nr)

        # TODO: Maybe we dont need this
        features_in_batches = features_in_batches.to(device)
        targets_in_batches = targets_in_batches.to(device)

        if task_nr != 0:
            predicted_target = inference(old_model, device, features_in_batches)
            previous_classes = classes_order[:(task_nr * tasks_nr)]
            targets_in_batches[:, previous_classes] = predicted_target[:, previous_classes]

        total_loss += train(model, device, loss_fn, optimizer, features_in_batches, targets_in_batches)

    accuracy, loss = get_accuracy(model, device, loss_fn, test_data_loader)

    print(f'train loss - {total_loss} | val loss - {loss} | accuracy - {accuracy}')

    scheduler.step()
