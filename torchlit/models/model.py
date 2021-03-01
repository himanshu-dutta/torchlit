import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericModel(nn.Module):
    r"""A generic model class which accepts different criterion functions as parameter::

        import torch.nn as nn
        import torch.nn.functional as F

        import torchlit.model as model

        class Net(model.Model):
            def __init__(self):
                super(Net, self).__init__(F.cross_entropy, record=True, verbose=True)
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))


    `record`:: `bool`:: `Optional`: Record the history over course of training.
    `verbose`:: `bool`:: `Optional`: Display information over epochs.
    """

    results = {
        "train": {"loss": []},
        "val": {"loss": [], "acc": []},
    }

    epoch = 0
    history = []

    def __init__(self, criterion, record: bool = True, verbose: bool = True):
        super(GenericModel, self).__init__()

        self.criterion = criterion
        self.record = record
        self.verbose = verbose

    def train_step(self, xb):
        data, labels = xb
        out = self(data)
        loss = self.criterion(out, labels)

        self.results["train"]["loss"].append(loss.cpu().item())

        return loss

    def val_step(self, xb):
        data, labels = xb
        out = self(data)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)

        self.results["val"]["loss"].append(loss.cpu().item())
        self.results["val"]["acc"].append(acc.cpu().item())

    def epoch_end(self):
        train_loss = torch.tensor(self.results["train"]["loss"]).mean().item()
        val_loss = torch.tensor(self.results["val"]["loss"]).mean().item()
        val_acc = torch.tensor(self.results["val"]["acc"]).mean().item()

        if self.record:
            self.history.append(
                {
                    "epoch": self.epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
        if self.verbose:
            print(
                f"Epoch [{self.epoch}]: train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}"
            )

        self.reset_results()
        self.epoch += 1

    def reset_results(self):
        self.results = {
            "train": {"loss": []},
            "val": {"loss": [], "acc": []},
        }

    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationModel(GenericModel):
    """
    Image Classification Model for Mulitclass Classification
    """

    def __init__(self, record=True, verbose=True):
        super().__init__(F.cross_entropy, record, verbose)
