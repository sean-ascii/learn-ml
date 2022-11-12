import torch
from torch import nn
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import data_loader
from util import utils


def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight, std=0.01)


batch_size = 256
train_iter, test_iter = data_loader.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction="none")

trainer = torch.optim.SGD(net.parameters(), lr = 0.1)

num_epochs = 10
utils.train_simple_classify(net, train_iter, test_iter, loss, num_epochs, trainer)
