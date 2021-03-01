import pytest

import torchlit
import torch.nn as nn
import torch.nn.functional as F


def test_createmodel():
    class Net(torchlit.Model):
        def __init__(self):
            super(Net, self).__init__(F.cross_entropy, record=True, verbose=True)
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    assert Net()
