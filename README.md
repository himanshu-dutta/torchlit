# torchlit

`torchlit` is an in progress collection of Pytorch utilities and thin wrappers which can be used for various purposes.

With every project, I intend to add functionalities that are fairly genralized to be put as a boilerplate for different utilities.

## Sample usage

```python
import torch.nn as nn
import torch.nn.functional as F

import torchlit

class Net(torchlit.Model):
    def __init__(self):
        super(Net, self).__init__(F.cross_entropy, record=True, verbose=True)
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

train_ds = Dataset()
val_ds = Dataset()

train_dl = DataLoader()
val_dl = DataLoader()

EPOCH = 100
model = Net()

for epoch in range(EPOCHS):
    for xb in train_dl:
        model.train_step(xb)

    for xb in val_dl:
        model.val_step(xb)

    model.epoch_end()
```
