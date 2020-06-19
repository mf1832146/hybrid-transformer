import torch.nn as nn
import torch
from torch.autograd import Variable
import math

from model.utils import clones


class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model, k, num_heads, num_features, dropout=0.0):
        """
                生成相对位置信息编码
                :param d_model: 词向量维度
                :param k: 相对位置窗口大小
                :param dropout:
                """
        super(RelativePositionEmbedding, self).__init__()

        self.d_model = d_model
        self.k = k
        self.num_features = num_features
        self.num_heads = num_heads

        assert self.num_heads % self.num_features == 0

        self.dropout = nn.Dropout(dropout)
        self.emb_list = clones(nn.Embedding(2*k+2, d_model * 2, padding_idx=0), num_features)

    def repeat_for_each_feature(self, emb):
        """
        :param emb: A Tensor with shape [batch_size, max_size, max_size, d_model]
        :return: A Tensor with shape [batch_size, num_head // num_features, max_size, max_size, d_model]
        """
        batch_size, max_size = emb.size(0), emb.size(1)
        emb = emb.repeat(1, 1, 1, self.num_heads // self.num_features)
        emb = emb.view(batch_size, max_size, max_size, -1, self.d_model)
        emb = emb.permute(0, 3, 1, 2, 4)
        return emb

    def forward(self, inputs):
        """inputs : A list of Tensor with shape [batch_size, max_size, max_size]"""
        assert self.num_features == len(inputs)

        k_emb_list = []
        v_emb_list = []

        for i, v in enumerate(inputs):
            batch_size, max_size = v.size(0), v.size(1)
            v = v.unsqueeze(3)
            position_emb = self.emb_list[i](inputs[i])
            position_emb = self.dropout(position_emb)
            position_emb = position_emb.view(batch_size, max_size, max_size, 2, self.d_model) * math.sqrt(self.d_model)
            k_emb, v_emb = [x.squeeze(3) for x in position_emb.split(1, dim=3)]

            k_emb = self.repeat_for_each_feature(k_emb)
            v_emb = self.repeat_for_each_feature(v_emb)

            k_emb_list.append(k_emb)
            v_emb_list.append(v_emb)

        k_emb_n = torch.cat(k_emb_list, dim=1)
        v_emb_n = torch.cat(v_emb_list, dim=1)

        return k_emb_n, v_emb_n


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)





