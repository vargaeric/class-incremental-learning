import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.1  # possible to have to change this to a smaller value


class iCaRL():
    def __init__(self):
        print('Initialize the iCaRL method!')

    def get_data_loaders(self, dataset):
        training_data = dataset(
            root='data',
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = dataset(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        )

        train_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE)
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        return train_data_loader, test_data_loader

    def get_device(self):
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def get_model(self, device):
        model = models.resnet18(weights=None).to(device)

        return model

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    def get_loss_fn(self):
        return torch.nn.CrossEntropyLoss()

    def train_loop(self, train_data_loader, device, model, loss_fn, optimizer):
        size = len(train_data_loader.dataset)

        model.train()

        for batch, (X, y) in enumerate(train_data_loader):
            X, y = X.to(device), y.to(device)

            # Make prediction and calculate the loss function
            pred = model(X)
            loss = loss_fn(pred, y)

            # Apply backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)

                print(f"loss: {round(loss, 5)}, [{round(current, 5)}/{round(size, 5)}]")

    def test_loop(self, test_data_loader, device, model, loss_fn):
        size = len(test_data_loader.dataset)
        batches_nr = len(test_data_loader)
        test_loss, correct = 0, 0

        model.eval()

        with torch.no_grad():
            for X, y in test_data_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= batches_nr
        correct /= size

        print(f"accuracy: {round(100 * correct, 5)}, avg loss: {round(test_loss, 5)}")

    def train(self):
        train_data_loader, test_data_loader = self.get_data_loaders(datasets.CIFAR100)
        device = self.get_device()
        model = self.get_model(device)
        optimizer = self.get_optimizer(model)
        loss_fn = self.get_loss_fn()

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}:")

            self.train_loop(train_data_loader, device, model, loss_fn, optimizer)
            self.test_loop(test_data_loader, device, model, loss_fn)

        print('Training finished!')
