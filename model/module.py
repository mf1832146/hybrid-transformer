import math, copy
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from model.attention import MultiHeadAttnRelative, MultiHeadAttn
from model.position_embedding import RelativePositionEmbedding, PositionalEncoding
from model.utils import gelu, subsequent_mask, clones, relative_mask


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, code_embed, nl_embed, generator, num_heads):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.code_embed = code_embed
        self.nl_embed = nl_embed
        self.generator = generator
        self.num_heads = num_heads

    def forward(self, inputs):
        code, relative_par_ids, relative_bro_ids, semantic_ids, nl, nl_mask = inputs

        encoder_code_mask = self.generate_code_mask(relative_par_ids, relative_bro_ids, semantic_ids)

        code_mask = (code == 0).unsqueeze(-2).unsqueeze(1)
        nl_embed = self.generate_nl_emb(nl)

        encoder_outputs = self.encode(code, relative_par_ids, relative_bro_ids, semantic_ids, encoder_code_mask)
        decoder_outputs, decoder_attn = self.decode(encoder_outputs, code_mask, nl_embed, nl_mask)
        return decoder_outputs, decoder_attn, encoder_outputs, nl_embed

    def encode(self, code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask):
        return self.encoder(self.code_embed(code), relative_par_ids, relative_bro_ids, semantic_ids, code_mask)

    def decode(self, memory, code_mask, nl_embed, nl_mask):
        return self.decoder(nl_embed, memory, code_mask, nl_mask)

    def generate_code_mask(self, relative_par_ids, relative_bro_ids, semantic_ids):
        relative_par_mask = relative_par_ids == 0
        relative_bro_mask = relative_bro_ids == 0
        semantic_mask = semantic_ids == 0

        code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], self.num_heads)
        return code_mask

    def generate_nl_emb(self, nl):
        return self.nl_embed(nl)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        output, attn = sublayer(self.norm(x))
        return x + self.dropout(output), attn


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x)))), None


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Encoder(nn.Module):
    def __init__(self, layer, N, relative_pos_emb, num_heads):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.relative_pos_emb = relative_pos_emb
        self.num_heads = num_heads

    def forward(self, code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask):
        relative_k_emb, relative_v_emb = self.relative_pos_emb([relative_par_ids, relative_bro_ids, semantic_ids])

        for layer in self.layers:
            code, attn = layer(code, code_mask, relative_k_emb, relative_v_emb)

        return self.norm(code)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, code, mask, relative_k_emb, relative_v_emb):
        code, attn = self.sublayer[0](code, lambda x: self.self_attn(x, x, x, mask, relative_k_emb, relative_v_emb))
        output, _ = self.sublayer[1](code, self.feed_forward)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x), attn


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x, nl_attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x, code_attn = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        output, _ = self.sublayer[2](x, self.feed_forward)
        return output, code_attn


class PointerGenerator(nn.Module):
    def __init__(self, d_model, nl_vocab_size, semantic_begin, max_simple_name_len, dropout):
        super(PointerGenerator, self).__init__()

        self.d_model = d_model
        self.nl_vocab_size = nl_vocab_size
        self.p_vocab = nn.Sequential(
            nn.Linear(self.d_model, self.nl_vocab_size - max_simple_name_len),
            nn.LogSoftmax(dim=-1)
        )
        self.p_gen = nn.Sequential(
            nn.Linear(3 * self.d_model, 1),
            nn.Sigmoid()
        )
        self.semantic_begin = semantic_begin
        self.log_soft_max = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_output, decoder_attn, memory, nl_embed, nl_convert, semantic_mask):
        """

        :param decoder_output: shape [batch_size, nl_len, d_model]
        :param decoder_attn: shape [batch_size, num_heads, nl_len, ast_len]
        :param memory: shape [batch_size, ast_len, d_model]
        :param nl_embed: shape [batch_size, nl_len, d_model]
        :param nl_convert: shape [batch_size, ast_len, max_simple_name_len]
        :param semantic_mask: shape [batch_size, ast_len]
        :return:
        """

        # shape [batch_size, nl_len, ast_len]
        decoder_attn = torch.sum(decoder_attn, dim=1)
        decoder_attn = decoder_attn.masked_fill(semantic_mask.unsqueeze(1) == 0, 0)

        p_vocab = self.p_vocab(decoder_output)  # shape [batch_size, nl_len, nl_vocab_size]
        context_vector = torch.matmul(decoder_attn, memory)  # shape [batch_size, nl_len, d_model]

        #  shape [batch_size, nl_len, 3 * d_model]
        total_state = torch.cat([context_vector, decoder_output, nl_embed], dim=-1)

        p_gen = self.p_gen(self.dropout(total_state))

        p_copy = 1 - p_gen

        # shape [batch_size, nl_len, max_simple_name_len]
        p_copy_ast = torch.matmul(decoder_attn, nl_convert) + 1e-9
        p_copy_ast = self.log_soft_max(p_copy_ast)

        if is_nan(p_copy_ast):
            print('p_copy_ast is null')
        if is_nan(p_vocab):
            print('p_vocab is null')
        if is_nan(torch.log(p_gen)):
            print('torch.log(p_gen) is null')
        if is_nan(torch.log(p_copy)):
            print('torch.log(p_copy) is null')

        p = torch.cat([p_vocab + torch.log(p_gen), p_copy_ast + torch.log(p_copy)], dim=-1)

        return p


def is_nan(inputs):
    return torch.sum(inputs != inputs) != 0


class Train(nn.Module):
    def __init__(self, model):
        super(Train, self).__init__()
        self.model = model

    def forward(self, inputs):
        code, relative_par_ids, relative_bro_ids, semantic_ids, semantic_convert_matrix, semantic_masks, nl, nl_mask = inputs
        decoder_outputs, decoder_attn, encoder_outputs, nl_embed = self.model.forward(
            (code, relative_par_ids, relative_bro_ids, semantic_ids, nl, nl_mask))
        out = self.model.generator(decoder_outputs, decoder_attn, encoder_outputs, nl_embed, semantic_convert_matrix, semantic_masks)
        return out


class GreedyEvaluate(nn.Module):
    def __init__(self, model,  max_nl_len, start_pos):
        super(GreedyEvaluate, self).__init__()
        self.model = model
        self.max_nl_len = max_nl_len
        self.start_pos = start_pos

    def forward(self, inputs):
        code, relative_par_ids, relative_bro_ids, semantic_ids, semantic_convert_matrix, semantic_masks, nl, nl_mask = inputs

        batch_size = code.size(0)

        encoder_code_mask = self.model.generate_code_mask(relative_par_ids, relative_bro_ids, semantic_ids)

        code_mask = (code == 0).unsqueeze(-2).unsqueeze(1)
        encoder_outputs = self.model.encode(code, relative_par_ids, relative_bro_ids, semantic_ids, encoder_code_mask)

        ys = torch.ones(batch_size, 1).fill_(self.start_pos).type_as(code.data)
        for i in range(self.max_nl_len - 1):
            nl_mask = Variable(subsequent_mask(ys.size(1)).type_as(code.data))
            nl_mask = (nl_mask == 0).unsqueeze(1)
            decoder_outputs, decoder_attn = self.model.decode(encoder_outputs,
                                                              code_mask,
                                                              self.model.generate_nl_emb(Variable(ys)),
                                                              nl_mask)

            prob = self.model.generator(decoder_outputs[:, -1].unsqueeze(1),
                                        decoder_attn[:, :, -1].unsqueeze(2),
                                        encoder_outputs,
                                        self.model.generate_nl_emb((Variable(ys)))[:, -1].unsqueeze(1),
                                        semantic_convert_matrix,
                                        semantic_masks)
            prob = prob.squeeze(1)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys,
                            next_word.unsqueeze(1).type_as(code.data)], dim=1)
        return ys


def make_model(code_vocab, nl_vocab, N=2, d_model=300, d_ff=512, k=5, h=6,
               num_features=3, max_simple_name_len=30, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttn(d_model, h)
    attn_relative = MultiHeadAttnRelative(d_model, h)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn_relative), c(ff), dropout), N,
                RelativePositionEmbedding(d_model // h, k, h, num_features, dropout), h),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        Embeddings(d_model, code_vocab),
        nn.Sequential(Embeddings(d_model, nl_vocab), c(position)),
        PointerGenerator(d_model, nl_vocab, h // num_features * (num_features-1), max_simple_name_len, dropout),
        num_heads=h
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == '__main__':
    make_model(50, 50)



