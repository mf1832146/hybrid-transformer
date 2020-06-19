import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from model.utils import clones


def _dot_product_attention_inner_relative(x, y, z, transpose):
    """
    :param x: is a Tensor with shape [batch_size, heads, length, length or depth].
    :param y: is a Tensor with shape [batch_size, heads, length, depth].
    :param z: is a Tensor with shape [batch_size, heads, length, length, depth].
    :param transpose: True if x[-1] is depth, else False.
    :return:
        A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size, heads, length, _ = x.size()

    #  xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = torch.matmul(x, y if not transpose else y.transpose(-2, -1))

    # x_t is [batch_size * heads * length, 1, length or depth]
    x_t = x.contiguous().view(batch_size * heads * length, -1).unsqueeze(1)
    # z_t is [batch_size * heads * length, length, depth]
    z_t = z.contiguous().view(batch_size * heads * length, length, -1)
    # xz_matmul is [batch_size * heads * length , 1, length or depth]
    xz_matmul = torch.matmul(x_t, z_t if not transpose else z_t.transpose(-2, -1))
    # xz_matmul_t is [batch_size, heads, length, length or depth]
    xz_matmul_t = xz_matmul.squeeze(1).view(batch_size, heads, length, -1)

    return xy_matmul + xz_matmul_t


def dot_product_attention_relative(q, k, v, bias, relative_k, relative_v, dropout):
    """
    :param q:
    :param k:
    :param v:
    :param bias:
    :param relative_k:
    :param relative_v:
    :param dropout:
    :return:
    """
    logits = _dot_product_attention_inner_relative(q, k, relative_k, True)
    if bias is not None:
        logits += bias
    # [batch, heads, length, length]
    weights = F.softmax(logits, -1)
    if dropout is not None:
        weights = dropout(weights)
    return weights, _dot_product_attention_inner_relative(weights, v, relative_v, False)


class MultiHeadAttn(nn.Module):
    def __init__(self, model_dim, head_count, dropout=0.1):
        super(MultiHeadAttn, self).__init__()
        self.model_dim = model_dim
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.head_count = head_count

        self.linear_layers = clones(nn.Linear(model_dim, model_dim), 4)
        self.dropout = nn.Dropout(dropout)
        self.soft_max = nn.Softmax(dim=-1)

    def split_heads(self, x):
        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, self.dim_per_head)\
            .transpose(1, 2).contiguous()

    def _combine_heads(self, x):
        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, query, key, value, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        query, key, value = \
            [self.split_heads(l(x))
             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.soft_max(scores)
        drop_attn = self.dropout(attn)
        context = self._combine_heads(torch.matmul(drop_attn, value))

        output = self.linear_layers[-1](context)

        return output, drop_attn


class MultiHeadAttnRelative(MultiHeadAttn):
    def __init__(self, model_dim, head_count, dropout=0.1):
        super().__init__(model_dim, head_count, dropout=dropout)

    def forward(self, query, key, value, mask=None, relative_k=None, relative_v=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        query, key, value = \
            [self.split_heads(l(x))
             for l, x in zip(self.linear_layers, (query, key, value))]

        query = query / math.sqrt(dim_per_head)

        # 2) Calculate and scale scores.
        if mask is not None:
            bias = mask.masked_fill(mask, -1e9)
        else:
            bias = None

        # do attention
        attn, context = dot_product_attention_relative(
            query, key, value,
            bias,
            relative_k, relative_v,
            self.dropout
        )

        # 3) Apply attention dropout and compute context vectors.
        # context ([batch, length, d_model])
        context = self._combine_heads(context)

        output = self.linear_layers[-1](context)

        return output, attn
