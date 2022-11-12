import torch
from torch import nn
from torch.nn import functional as F
import rnn_framework

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import data_loader
from util import utils

batch_size, num_steps = 32, 35
train_iter, vocab = data_loader.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

device=utils.try_gpu()
net = rnn_framework.RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
# print(rnn_framework.predict_rnn('time traveller', 10, net, vocab, device))
num_epochs, lr = 500, 1
rnn_framework.train_rnn(net, train_iter, vocab, lr, num_epochs, device)