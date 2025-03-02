import torch
from torch import nn
from utils import *

class EncoderBlock(nn.Module):
  '''Transformer编码器块'''
  def __init__(self, key_size, query_size, value_size, num_hiddens,
               norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
               dropout, use_bias=False, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)

    self.attention = MultiHeadAttention(
        key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
    self.add_norm1 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(
        ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.add_norm2 = AddNorm(norm_shape, dropout)

  def forward(self, X, valid_lens):
    Y = self.add_norm1(X, self.attention(X, X, X, valid_lens))
    return self.add_norm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
  '''Transformer编码器'''
  def __init__(self, vocab_size, key_size, query_size, value_size,
               num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
               num_heads, num_layers, dropout, use_bias=False, **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = PositionEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i),
                           EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                        norm_shape, ffn_num_input, ffn_num_hiddens,
                                        num_heads, dropout, use_bias))
      
  def forward(self, X, valid_lens, *args):
    # 因为位置编码值在-1和1之间， 因此嵌入值乘以嵌入维度的平方根进行缩放，
    # 然后再与位置编码相加
    # why: https://zhuanlan.zhihu.com/p/442509602
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self.attention_weights = [None] * len(self.blks)
    for i, blk in enumerate(self.blks):
      X = blk(X, valid_lens)
      self.attention_weights[i] = blk.attention.attention.attention_weights
    return X

