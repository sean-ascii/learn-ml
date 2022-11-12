import torch
from torch import nn
import math
from torch.nn import functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import utils

def predict_rnn(prefix, num_preds, net, vocab, device):
  """在prefix后面生成字符"""
  state = net.begin_state(batch_size=1, device=device)
  outputs = [vocab[prefix[0]]]
  get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
  for y in prefix[1:]: # 预热期，只更新模型和隐状态，不进行预测
    _, state = net(get_input(), state)
    outputs.append(vocab[y])
  for _ in range(num_preds): # 预测num_preds步
    y, state = net(get_input(), state)
    outputs.append(int(y.argmax(dim=1).reshape(1)))
  return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
  """裁剪梯度"""
  if isinstance(net, nn.Module):
    params = [p for p in net.parameters() if p.requires_grad]
  else:
    params = net.params
  norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
  if norm > theta:
    for param in params:
      param.grad[:] *= theta / norm

def train_epoch_rnn(net, train_iter, loss, updater, device, use_random_iter):
  """训练网络一个迭代周期"""
  state = None
  metric = utils.Accumulator(2)
  for X, Y in train_iter:
    if state is None or use_random_iter:
      # 在第一次迭代或使用随机抽样时初始化state
      state = net.begin_state(batch_size=X.shape[0], device=device)
    else:
      if isinstance(net, nn.Module) and not isinstance(state, tuple):
        # state对于nn.GRU是个张量
        state.detach_()
      else:
        # state对于nn.LSTM是元组
        for s in state:
          s.detach_()
    y = Y.T.reshape(-1)
    # print(f'X shape: {X.shape}, Y shape: {Y.shape}, y shape: {y.shape}')
    X, y = X.to(device), y.to(device)
    y_hat, state = net(X, state)
    # print(f'y_hat shape: {y_hat.shape}')
    # print(y_hat[0])
    l = loss(y_hat, y.long()).mean()
    if isinstance(updater, torch.optim.Optimizer):
      updater.zero_grad()
      l.backward()
      grad_clipping(net, 1)
      updater.step()
    else:
      l.backward()
      grad_clipping(net, 1)
      updater(batch_size=1)
    metric.add(l * y.numel(), y.numel())
  return math.exp(metric[0] / metric[1])

def train_rnn(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
  """训练模型"""
  loss = nn.CrossEntropyLoss()
  # 初始化
  if isinstance(net, nn.Module):
    updater = torch.optim.SGD(net.parameters(), lr)
  else:
    updater = lambda batch_size: utils.sgd(net.params, lr, batch_size)
  predict = lambda prefix: predict_rnn(prefix, 50, net, vocab, device)
  # 训练和预测
  for epoch in range(num_epochs):
    ppl = train_epoch_rnn(
        net, train_iter, loss, updater, device, use_random_iter)
    if (epoch + 1) % 10 == 0:
      print(predict('time traveller'))
  print(f'困惑度 {ppl:.1f}, {str(device)}')
  print(predict('time traveller'))
  print(predict('traveller'))

class RNNModelScratch:
  """从零开始实现的循环神经网络模型"""
  def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
    self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
    self.params = get_params(vocab_size, num_hiddens, device)
    self.init_state, self.forward_fn = init_state, forward_fn

  def __call__(self, X, state):
    X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
    return self.forward_fn(X, state, self.params)

  def begin_state(self, batch_size, device):
    return self.init_state(batch_size, self.num_hiddens, device)

class RNNModel(nn.Module):
  """循环神经网络模型"""
  def __init__(self, rnn_layer, vocab_size, **kwargs):
    super(RNNModel, self).__init__(**kwargs)
    self.rnn = rnn_layer
    self.vocab_size = vocab_size
    self.num_hiddens = self.rnn.hidden_size
    # 如果RNN是双向的，num_directions是2，否则是1
    if not self.rnn.bidirectional:
      self.num_directions = 1
      self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
    else:
      self.num_directions = 2
      self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

  def forward(self, inputs, state):
    X = F.one_hot(inputs.T.long(), self.vocab_size)
    X = X.to(torch.float32)
    Y, state = self.rnn(X, state)
    # 全连接层首先将Y的形状改为（时间步数*批量大小，隐藏单元数）
    # 它的输出形状是（时间步数*批量大小，词表大小）
    output = self.linear(Y.reshape((-1, Y.shape[-1])))
    return output, state

  def begin_state(self, device, batch_size=1):
    if not isinstance(self.rnn, nn.LSTM):
      # nn.GRU以张量作为隐状态
      return torch.zeros((self.num_directions * self.rnn.num_layers,
                          batch_size, self.num_hiddens), device=device)
    else:
      # nn.LSTM以元组作为隐状态
      return (torch.zeros((self.num_directions * self.rnn.num_layers,
                           batch_size, self.num_hiddens), device=device),
              torch.zeros((self.num_directions * self.rnn.num_layers,
                           batch_size, self.num_hiddens), device=device))