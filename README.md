# torchlit

`torchlit` is an in progress collection of Pytorch utilities and thin wrappers which can be used for various purposes.

With every project, I intend to add functionalities that are fairly genralized to be put as a boilerplate for different utilities.

## Sample usage

```
!pip install torchlit --q
```

```
    |████████████████████████████████| 911kB 5.4MB/s
    |████████████████████████████████| 102kB 7.3MB/s
    |████████████████████████████████| 81kB 6.7MB/s
    |████████████████████████████████| 7.6MB 9.3MB/s
    |████████████████████████████████| 81kB 7.4MB/s
    |████████████████████████████████| 102kB 9.5MB/s
```

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchlit
```

```Python
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
```

```
    Net(
      (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(6, 12, kernel_size=(3, 3), stride=(1, 1))
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (lin): Linear(in_features=184512, out_features=10, bias=True)
    )
```

```Python
train_ds = [(x, y) for x,y in zip(torch.randn((10, 3, 128, 128)), torch.randint(0, 10, (10,)))]
val_ds = [(x,y) for x,y in zip(torch.randn((3, 3, 128, 128)), torch.randint(0, 10, (3,)))]

train_dl = DataLoader(train_ds)
val_dl = DataLoader(val_ds)
```

```Python
EPOCHS = 10
model = Net()

for epoch in range(EPOCHS):
    for xb in train_dl:
        model.train_step(xb)

    for xb in val_dl:
        model.val_step(xb)

    model.epoch_end()
```

```

    Epoch [0]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [1]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [2]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [3]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [4]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [5]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [6]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [7]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [8]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
    Epoch [9]: train_loss: 2.3065271377563477, val_loss: 2.3060548305511475, val_acc: 0.0
```
