import collections
import enum
from lib2to3.pgen2 import token
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import re
from pathlib import Path
import random

def load_array(data_arrays, batch_size, is_train=True):
  # 构造pytorch数据迭代器
  dataset = data.TensorDataset(*data_arrays)
  return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels):
  '''将索引转化为文本标签'''
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
  figsize = (num_cols * scale, num_rows * scale)
  _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    if torch.is_tensor(img):
      ax.imshow(img.numpy())
    else:
      ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  plt.show()

def get_dataloader_workers():
  '''使用4个进程来读取数据'''
  return 4

def load_data_fashion_mnist(batch_size, resize=None):
  '''下载Fashion-MNIST数据集，并加载到内存中'''
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)
  # 图像加载，并通过ToTensor将图像像素数据转化为0-1之间的浮点数值
  mnist_train = torchvision.datasets.FashionMNIST(
      root="~/code/learn/python/learn-ml/data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(
      root="~/code/learn/python/learn-ml/data", train=False, transform=trans, download=True)
  # print(len(mnist_train), len(mnist_test))
  # print(mnist_train[0][0].shape)

  # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
  # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
  return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                          num_workers=get_dataloader_workers()),
          data.DataLoader(mnist_test, batch_size, shuffle=False,
                          num_workers=get_dataloader_workers()))

def read_time_machine():
  file_path = str(Path(__file__).resolve().parents[1]) + "/data/timemachine.txt"
  with open(file_path, 'r') as f:
    lines = f.readlines()
  return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
  """将文本行拆分成单词或字符词元"""
  if token == 'word':
    return [line.split() for line in lines]
  elif token == 'char':
    return [list(line) for line in lines]
  else:
    print('错误：未知词元类型：' + token)

class Vocab:
  """文本词表"""
  def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
    if tokens is None:
      tokens = []
    if reserved_tokens is None:
      reserved_tokens = []
    counter = count_corpus(tokens)
    self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    self.idx_to_token = ['<unk>'] + reserved_tokens
    self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
    for token, freq in self._token_freqs:
      if freq < min_freq:
        break
      if token not in self.token_to_idx:
        self.idx_to_token.append(token)
        self.token_to_idx[token] = len(self.idx_to_token) - 1

  def __len__(self):
    return len(self.idx_to_token)

  def __getitem__(self, tokens):
    if not isinstance(tokens, (list, tuple)):
      return self.token_to_idx.get(tokens, self.unk)
    return [self.__getitem__(token) for token in tokens]

  def to_tokens(self, indices):
    if not isinstance(indices, (list, tuple)):
      return self.idx_to_token[indices]
    return [self.idx_to_token[index] for index in indices]

  @property
  def unk(self):
    return 0

  @property
  def token_freqs(self):
    return self._token_freqs

def count_corpus(tokens):
  """统计词元频率"""
  if len(tokens) == 0 or isinstance(tokens[0], list):
    # 将词元列表展平成一个列表
    tokens = [token for line in tokens for token in line]
  return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
  """返回时光机器数据集的词元索引列表和词表"""
  lines = read_time_machine()
  tokens = tokenize(lines, 'char')
  vocab = Vocab(tokens)
  # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落
  # 所以将所有的文本行展平到一个列表中
  corpus = [vocab[token] for line in tokens for token in line]
  if max_tokens > 0:
    corpus = corpus[:max_tokens]
  return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
  """使用随机抽样生成一个小批量子序列"""
  # 从随机偏移量开始对序列进行分区，随机范围包括num_steps - 1
  corpus = corpus[random.randint(0, num_steps - 1):]
  # 减去1，因为我们需要考虑标签
  num_subseqs = (len(corpus) - 1) // num_steps
  # 长度为num_steps的子序列的起始索引
  initial_indices =  list(range(0, num_subseqs * num_steps, num_steps))
  # 在随机抽样的迭代过程中，来自两个相邻的、随机的、小批量的子序列不一定在原始序列上相邻
  random.shuffle(initial_indices)

  def data(pos):
    return corpus[pos: pos + num_steps]

  num_batches = num_subseqs // batch_size
  for i in range(0, batch_size * num_batches, batch_size):
    # 在这里，initial_indices包含子序列的随机起始索引
    initial_indices_per_batch = initial_indices[i: i + batch_size]
    X = [data(j) for j in initial_indices_per_batch]
    Y = [data(j + 1) for j in initial_indices_per_batch]
    yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
  """使用顺序分区生成一个小批量子序列"""
  # 从随机偏移量开始划分序列
  offset = random.randint(0, num_steps)
  num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
  Xs = torch.tensor(corpus[offset: offset + num_tokens])
  Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
  Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, - 1)
  num_batches = Xs.shape[1] // num_steps
  for i in range(0, num_steps * num_batches, num_steps):
    X = Xs[:, i: i + num_steps]
    Y = Ys[:, i: i + num_steps]
    yield X, Y

class SeqDataLoader:
  """加载序列数据的迭代器"""
  def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
    if use_random_iter:
      self.data_iter_fn = seq_data_iter_random
    else:
      self.data_iter_fn = seq_data_iter_sequential
    self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
    self.batch_size, self.num_steps = batch_size, num_steps

  def __iter__(self):
    return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
  """返回时光机器数据集的迭代器和词表"""
  data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
  return data_iter, data_iter.vocab


# # corpus, vocab = load_corpus_time_machine()
# # print(len(corpus), len(vocab))
# my_seq = list(range(35))
# # for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
# #   print('X: ', X, '\nY: ', Y)
# for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
#   print('X: ', X, '\nY: ', Y)