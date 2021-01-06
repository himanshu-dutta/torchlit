import pytest

import random

import torch
from .pytorch import ResNet50MLC


def test_createmodel():
    assert ResNet50MLC(random.randint(1, 100))
