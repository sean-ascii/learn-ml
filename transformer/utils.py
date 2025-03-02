import torch
from torch import nn
import math

def dropout_layer(X, dropout):
  assert 0 <= dropout <= 1
  # 在本情况中，所有元素都被丢弃
  if dropout == 1:
    return torch.zeros_like(X)
  # 在本情况中，所有元素都被保留
  if dropout == 0:
    return X
  mask = (torch.rand(X.shape) > dropout).float()
  return mask * X / (1.0 - dropout)

def sequence_mask(X, valid_len, value=0):
  '''在序列中屏蔽不相关的项'''
  max_len = X.size(1)
  # [None, :]将list转变为1 * n的二维数组, [:, None]将list转变为n * 1的二维数组，
  # 然后利用广播机制构造出了mask矩阵
  mask = torch.arange((max_len), dtype=torch.flaot32,
                      device=X.device)[None, :] < valid_len[:, None]
  X[~mask] = value
  return X

def masked_softmax(X, valid_lens):
  '''通过在最后一个轴上掩蔽元素来执行softmax操作'''
  # X: 3D张量， valid_lens：1D或2D张量
  if valid_lens is None:
    return nn.functional.softmax(X, dim=-1)
  else:
    shape = X.shape
    if valid_lens.dim() == 1:
      valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
      valid_lens = valid_lens.reshape(-1)
    # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而使其softmax输出为0
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductionAttention(nn.Module):
  '''缩放点积注意力'''
  def __init__(self, dropout, **kwargs):
    super(DotProductionAttention, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)

  # queries的形状：(batch_size, 查询的个数， d)
  # keys的形状：(batch_size, “键值对”的个数， d)
  # values的形状：(batch_size, "键值对"的个数， d)
  # valid_lens的形状：(batch_size,)或者(batch_size, 查询的个数)
  def forward(self, queries, keys, values, valid_lens=None):
    d = queries.shape[-1]
    # 设置transpose_b=True为了交换keys的最后两个维度
    scores = torch.bmm(queries, keys.tranpose(1, 2)) / math.sqrt(d)
    self.attention_weights = masked_softmax(scores, valid_lens)
    return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
  '''为了多注意力头的并行计算而变换形状'''
  # 输入X的形状：(batch_size, 查询或者“键值对”的个数，num_hiddens)
  # 输出X的形状：(batch_size, 查询或者“键值对”的个数，num_heads, num_hiddens/num_heads)
  X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

  # 输出X的形状：(batch_size, num_heads, 查询或者“键值对”的个数，num_hiddens/num_heads)
  X = X.permute(0, 2, 1, 3)

  # 最终输出的形状：(batch_size*num_heads, 查询或者“键值对”的个数， num_hiddens/num_heads)
  return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
  '''逆转transpose_qkv函数的操作'''
  X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
  X = X.permute(0, 2, 1, 3)
  return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
  '''多头注意力'''
  def __init__(self, key_size, query_size, value_size, num_hiddens,
               num_heads, dropout, bias=False, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.attention = DotProductionAttention(dropout)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
    self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
    self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

  def forward(self, queris, keys, values, valid_lens):
    # queries, keys, values的形状:
    #   (batch_size, 查询或者“键值对”的个数， num_hiddens)
    # valid_lens的形状:
    #   (batch_size,) 或 (batch_size, 查询的个数)

    # 经过变换后，输出的queries, keys, values的形状：
    #   (batch_size * num_heads, 查询或者“键值对”的个数，num_hiddens/num_heads)
    queris = transpose_qkv(self.W_q(queris), self.num_heads)
    keys = transpose_qkv(self.W_k(keys), self.num_heads)
    values = transpose_qkv(self.W_v(values), self.num_heads)

    if valid_lens is not None:
      # 在轴0，将第一项(标量或者矢量)复制num_heads次，
      # 然后如此复制第二项，然后诸如此类
      valid_lens = torch.repeat_interleave(
          valid_lens, repeats=self.num_heads, dim=0)

    # output的形状: (batch_size*num_heads, 查询的个数, num_hiddens/num_heads)
    output = self.attention(queris, keys, values, valid_lens)

    # output_concat的形状：(batch_size, 查询的个数，num_hiddens)
    output_concat = transpose_output(output, self.num_heads)
    return self.W_o(output_concat)

class PositionWiseFFN(nn.Module):
  '''基于位置的前馈网络'''
  def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
    super(PositionWiseFFN, self).__init__(**kwargs)
    self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
    self.relu = nn.ReLU()
    self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

  def forward(self, X):
    return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
  '''残差连接后进行层规范化'''
  def __init__(self, normalized_shape, dropout, **kwargs):
    super(AddNorm, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, X, Y):
    return self.ln(self.dropout(Y) + X)


# p_{i, 2j} = sin(i / (10000^{2j/d}))
# p_{i, 2j + 1} = cos(i / (10000^{2j/d}))
class PositionEncoding(nn.Module):
  '''位置编码'''
  def __init__(self, num_hiddens, dropout, max_len=1000):
    super(PositionEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)
    # 创建一个足够长的P
    self.P = torch.zeros((1,  max_len, num_hiddens))
    X = torch.arange(max_len, dtype=torch.float32).reshape(
        -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
    self.P[:, :, 0::2] = torch.sin(X)
    self.P[:, :, 1::2] = torch.cos(X)

  def forward(self, X):
    X = X + self.P[:, :X.shape[1], :].to(X.device)
    return self.dropout(X)