from cProfile import label
from turtle import title
import torch
from torch import nn
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import data_loader
from util import utils

"""自回归模型"""

def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.xavier_normal_(m.weight)

def get_net():
  net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
  net.apply(init_weights)
  return net

def train(net, train_iter, loss, epochs, lr):
  trainer = torch.optim.Adam(net.parameters(), lr)
  for epoch in range(epochs):
    for X, y in train_iter:
      trainer.zero_grad()
      l = loss(net(X), y)
      l.sum().backward()
      trainer.step()
    print(f'epoch {epoch + 1}, '
          f'loss: {utils.evaluate_loss(net, train_iter, loss):f}')

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# plt.plot(time, x)
# plt.xlabel('time')
# plt.ylabel('x')
# plt.xlim(1, 1000)
# plt.show()

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
  features[:, i] = x[i : T - tau + i]
labels = x[tau : ].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = data_loader.load_array((features[:n_train], labels[:n_train]),
                                    batch_size, is_train=True)

loss = nn.MSELoss(reduction='none')
net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
plt.plot(time, x.detach().numpy(), label='data')
plt.plot(time[tau:], onestep_preds.detach().numpy(), label='1-step preds')
plt.legend()
plt.show()
