# torchlit

`torchlit` is an in progress collection of Pytorch utilities and thin wrappers which can be used for various purposes.

With every project, I intend to add functionalities that are fairly genralized to be put as a boilerplate for different utilities.

## Sample usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchlit

class Net(torchlit.Model):
    def __init__(self):
        super(Net, self).__init__(F.cross_entropy, record=True, verbose=True)
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(184512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        return self.lin(x)

model = Net()
model

train_ds = [(x, y) for x,y in zip(torch.randn((10, 3, 128, 128)), torch.randint(0, 10, (10,)))]
val_ds = [(x,y) for x,y in zip(torch.randn((3, 3, 128, 128)), torch.randint(0, 10, (3,)))]

train_dl = DataLoader(train_ds)
val_dl = DataLoader(val_ds)

EPOCHS = 10


for epoch in range(EPOCHS):
    for xb in train_dl:
        model.train_step(xb)

    for xb in val_dl:
        model.val_step(xb)

    model.epoch_end()
```
