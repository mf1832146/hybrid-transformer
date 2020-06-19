import numpy as np

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from torch import nn
from torch.autograd import Variable

from train.evaluation import batch_bleu, batch_meteor, batch_rouge_l
from model.utils import subsequent_mask


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        x = x.contiguous().view(-1, x.size(-1))
        predicts = target[:, 1:]
        ntokens = (predicts != 0).data.sum()
        target = target.contiguous().view(-1)
        assert x.size(1) == self.size

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False)) / ntokens


class BLEU4(Metric):
    def __init__(self, id2nl, output_transform=lambda x: x, device=None):
        super(BLEU4, self).__init__(output_transform, device=device)
        self._id2nl = id2nl

    @reinit__is_reduced
    def reset(self):
        self._bleu_scores = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        scores = batch_bleu(y, y_pred, self._id2nl, self._num_examples)
        self._bleu_scores += np.sum(scores)
        self._num_examples += len(scores)

    @sync_all_reduce("_bleu_scores", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BLEU4 must have "
                                     "at least one example before it can be computed.")
        return self._bleu_scores / self._num_examples


class Rouge(Metric):
    def __init__(self, id2nl, output_transform=lambda x: x, device=None):
        super(Rouge, self).__init__(output_transform, device=device)
        self._id2nl = id2nl

    @reinit__is_reduced
    def reset(self):
        self._rouge_scores = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        scores = batch_rouge_l(y, y_pred, self._id2nl, self._num_examples)
        self._rouge_scores += np.sum(scores)
        self._num_examples += len(scores)

    @sync_all_reduce("_rouge_scores", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BLEU4 must have "
                                     "at least one example before it can be computed.")
        return self._rouge_scores / self._num_examples


class Meteor(Metric):
    def __init__(self, id2nl, output_transform=lambda x: x, device=None):
        super(Meteor, self).__init__(output_transform, device=device)
        self._id2nl = id2nl

    @reinit__is_reduced
    def reset(self):
        self._meteor_scores = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        scores = batch_meteor(y, y_pred, self._id2nl)
        self._meteor_scores += np.sum(scores)
        self._num_examples += len(scores)

    @sync_all_reduce("_meteor_scores", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BLEU4 must have "
                                     "at least one example before it can be computed.")
        return self._meteor_scores / self._num_examples


class MyLoss(Metric):
    def __init__(self, nl_vocab_size, output_transform=lambda x: x, device=None):
        super(MyLoss, self).__init__(output_transform, device=device)
        """不使用平滑"""
        self.criterion = LabelSmoothing(size=nl_vocab_size,
                                        padding_idx=0, smoothing=0)

    @reinit__is_reduced
    def reset(self):
        self._losses = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        x, y = output
        predicts = y[:, 1:]
        ntokens = (predicts != 0).data.sum()

        loss = self.criterion(x, y)

        self._losses += loss * ntokens
        self._num_examples += ntokens

    @sync_all_reduce("_losses", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._losses / self._num_examples


class Batch:
    def __init__(self,
                 code,
                 re_par_ids,
                 re_bro_ids,
                 comments=None,
                 pad=0):
        self.code = code
        # code_mask用于解码时用
        self.code_mask = (code != pad).unsqueeze(-2)
        self.re_par_ids = re_par_ids
        self.re_bro_ids = re_bro_ids
        if comments is not None:
            self.comments = comments[:, :-1]
            self.predicts = comments[:, 1:]
            self.comment_mask = self.make_std_mask(self.comments, pad)
            # 训练时的有效预测个数
            self.ntokens = (self.predicts != pad).data.sum()

    @staticmethod
    def make_std_mask(comment, pad):
        comment_mask = (comment != pad).unsqueeze(-2)
        tgt_mask = comment_mask & Variable(
            subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
        return tgt_mask


def ast_vocab_2_nl_vocab(id2ast, id2nl, unk_id=1):
    """
    :param id2ast: ast词汇表
    :param id2nl: nl词汇表
    :param unk_id: 未收纳词编号
    :return: ast2nl
    """
    nl2id = {v: k for k, v in id2ast.items()}
    ast2nl = torch.zeros(len(id2ast), len(id2nl))
    for ast_id in id2ast:
        ast_value = id2ast[ast_id]
        if ast_value in nl2id.keys():
            ast2nl[ast_id][nl2id[ast_value]] = 1
        else:
            ast2nl[ast_id][unk_id] = 1

    return ast2nl

