"""
Contains class for training and testing a PyTorch model
"""

import torch
from tqdm.auto import tqdm

class Train():
    """
    Train and Test model for multiple epochs.

    Turns a model to train and test mode and then runs forward pass, loss calculation, and parameter optimization.

    Args:
        model: PyTorch model to train
        epochs: Number of iterations
        optimizer: A PyTorch optimizer to help minimize the loss function
        loss_fn: PyTorch loss function to minimize
        train_dataloader: Train data
        test_dataloader: Test data
        device: A target device to compute on (e.g. "cuda" or "cpu")
    """
    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 device: torch.device
                 ):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def train_steps(self):
        self.model.train()

        train_loss, train_acc = 0, 0

        for batch, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            output = self.model(x)

            # Calculate loss
            loss = self.loss_fn(output, y)
            train_loss += loss.item()

            # Optimizer steps
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            preds = torch.argmax(output, dim=1)
            train_acc += (preds == y).sum().item() / len(y)

        # Average loss and accuracy per epoch
        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        return train_acc, train_loss, self.model

    def test_step(self,model):
        self.model.eval()

        test_loss, test_acc = 0, 0

        with torch.no_grad():
            for batch, (x, y) in enumerate(self.test_dataloader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                output = model(x)

                # Calculate loss
                loss = self.loss_fn(output, y)
                test_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(output, dim=1)
                test_acc += (preds == y).sum().item() / len(y)

        # Average loss and accuracy
        test_loss /= len(self.test_dataloader)
        test_acc /= len(self.test_dataloader)
        return test_acc, test_loss, model

    def model_evaluation(self):
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

        for epoch in tqdm(range(self.epochs)):
            print(f"\nEpoch {epoch+1}/{self.epochs} Training...")

            train_acc, train_loss, model = self.train_steps()
            test_acc, test_loss, model = self.test_step(model=model)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

            # Update results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        return results ,model
