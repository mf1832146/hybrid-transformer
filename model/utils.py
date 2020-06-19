import torch
import math
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(comment, pad):
    comment_mask = (comment != pad).unsqueeze(-2)
    tgt_mask = comment_mask & Variable(
        subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
    return tgt_mask


def pad_seq(data_list, max_len):
    data = torch.zeros(max_len)
    for i in range(min(max_len, len(data_list))):
        data[i] = data_list[i]
    return data


def relative_mask(masks, num_heads):
    """masks : A List of Tensor with shape [batch_size, seq_len, seq_len]"""
    num_features = len(masks)

    assert num_heads % num_features == 0

    output = []

    for m in masks:
        m = m.unsqueeze(1).repeat(1, num_heads // num_features, 1, 1)
        output.append(m)

    return torch.cat(output, dim=1)


def get_semantic_attn_weights(attn, head_begin):
    """
    sum the semantic attn weights for pointer generator
    :param attn: shape [batch_size, num_heads, nl_len, ast_len]
    :param head_begin: is an integer that indicates where the semantic attn begins
    :return: semantic_attn shape [batch_size, nl_len, ast_len]
    """
    batch_size, num_heads = attn.size(0), attn.size(1)
    semantic_attn = attn[:, head_begin:, :, :]
    semantic_attn = torch.sum(semantic_attn, dim=1)

