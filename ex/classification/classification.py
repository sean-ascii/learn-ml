from audioop import cross
from cProfile import label
from unicodedata import name
import torch
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import data_loader
from util import utils

"""基于线性模型做分类"""

batch_size = 256
train_iter, test_iter = data_loader.load_data_fashion_mnist(batch_size)

# for X, y in train_iter:
#   print(X.shape, X.dtype, y.shape, y.dtype)
#   break

num_inputs = 784 #28 * 28
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
  X_exp = torch.exp(X)
  partition = X_exp.sum(1, keepdim=True)
  return X_exp / partition  # 应用广播机制

def net(X):
  return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
  return -torch.log(y_hat[range(len(y_hat)), y])

lr = 0.1

def updater(batch_size):
  return utils.sgd([W, b], lr, batch_size)

num_epochs = 10
utils.train_simple_classify(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict(net, test_iter, n=6):
  for X, y in test_iter:
    break
  trues = data_loader.get_fashion_mnist_labels(y)
  preds = data_loader.get_fashion_mnist_labels(net(X).argmax(axis=1))
  titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
  data_loader.show_images(
      X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict(net, test_iter)
