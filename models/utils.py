import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _, labels = torch.max(labels, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)