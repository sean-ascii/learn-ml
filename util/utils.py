import torch
import matplotlib.pyplot as plt

class Accumulator:
  def __init__(self, n):
    self.data = [0.0] * n

  def add(self, *args):
    self.data = [a + float(b) for a, b in zip (self.data, args)]

  def reset(self):
    self.data = [0.0] * len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


# learning algorithm, 小批量随机梯度下降
def sgd(params, lr, batch_size):
  with torch.no_grad():
    for param in params:
      param -= lr * param.grad / batch_size
      param.grad.zero_()

def accuracy(y_hat, y):
  """计算预测正确的数量"""
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = y_hat.argmax(axis = 1)
  cmp = y_hat.type(y.dtype) == y
  return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
  """计算在指定数据集上模型的精度"""
  if isinstance(net, torch.nn.Module):
    net.eval()
  metric = Accumulator(2)
  with torch.no_grad():
    for X, y in data_iter:
      metric.add(accuracy(net(X), y), y.numel())
  return metric[0] / metric[1]

def evaluate_loss(net, data_iter, loss):
  """评估给定数据集上模型的损失"""
  metric = Accumulator(2)
  for X, y in data_iter:
    out = net(X)
    y = y.reshape(out.shape)
    l = loss(out, y)
    metric.add(l.sum(), l.numel())
  return metric[0] / metric[1]

def train_epoch_simple_classify(net, train_iter, loss, updater):
  if isinstance(net, torch.nn.Module):
    net.train()
  metric = Accumulator(3)
  for X, y in train_iter:
    y_hat = net(X)
    l = loss(y_hat, y)
    if isinstance(updater, torch.optim.Optimizer):
      updater.zero_grad()
      l.mean().backward()
      updater.step()
    else:
      l.sum().backward()
      updater(X.shape[0])
    metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
  return metric[0] / metric[2], metric[1] / metric[2]

def train_simple_classify(net, train_iter, test_iter, loss, num_epochs, updater):
  result_train_loss = []
  result_train_acc = []
  result_test_acc = []
  for epoch in range(num_epochs):
    train_metrics = train_epoch_simple_classify(net, train_iter, loss, updater)
    test_acc = evaluate_accuracy(net, test_iter)
    result_train_loss.append(train_metrics[0])
    result_train_acc.append(train_metrics[1])
    result_test_acc.append(test_acc)
  plt.plot(result_train_loss, label="train loss")
  plt.plot(result_train_acc, label="train acc")
  plt.plot(result_test_acc, label="test acc")
  plt.legend()
  plt.show()

def try_gpu(i=0):
  """如果存在，则返回gpu(i)，否则返回cpu()"""
  if torch.cuda.device_count() >= i + 1:
    return torch.device(f'cuda:{i}')
  return torch.device('cpu')

def try_all_gpus():
  """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
  devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
  return devices if devices else [torch.device('cpu')]

# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())