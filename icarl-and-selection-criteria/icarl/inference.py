from torch import no_grad


def inference(model, device, features_in_batches, return_features=False):
    model.eval()

    with no_grad():
        features_in_batches = features_in_batches.to(device)
        output_or_features = model(features_in_batches, return_features)

    return output_or_features
