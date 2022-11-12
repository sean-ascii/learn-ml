import torch
from torch import nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import data_loader
from util import utils

"""使用MLP做简单的图像分类"""

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = data_loader.load_data_fashion_mnist(batch_size)
utils.train_simple_classify(net, train_iter, test_iter, loss, num_epochs, trainer)