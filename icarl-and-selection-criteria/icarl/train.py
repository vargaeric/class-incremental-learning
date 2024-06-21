def train(model, device, criterion, optimizer, features_in_batches, targets_in_batches):
    model.train()
    model.zero_grad()

    features_in_batches = features_in_batches.to(device)
    targets_in_batches = targets_in_batches.to(device)
    output = model(features_in_batches)
    loss = criterion(output, targets_in_batches)

    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()
