import torch
from torch import nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import data_loader
from util import utils
sys.path.append(str(Path(__file__).resolve().parents[1]))
import rnn_framework

batch_size, num_steps = 32, 35
train_iter, vocab = data_loader.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, utils.try_gpu()

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = rnn_framework.RNNModel(gru_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 1
rnn_framework.train_rnn(model, train_iter, vocab, lr, num_epochs, device)