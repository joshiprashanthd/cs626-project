import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import DevanagariNormalizer, OriyaNormalizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

from transformers import AutoTokenizer

import io, os, re, random, time, math


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, device):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.toquery = nn.Linear(hid_dim, hid_dim)
        self.tokey = nn.Linear(hid_dim, hid_dim)
        self.tovalue = nn.Linear(hid_dim, hid_dim)

        self.ffn = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(0.2)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query = key = value = [batch_size, sequence_len, hid_dim]
        batch_size = query.shape[0]

        q = self.toquery(query)
        k = self.tokey(key)
        v = self.tovalue(value)

        # q = k = v = [batch_size, sequence_len, hid_dim]

        q = q.view((batch_size, -1, self.n_heads, self.head_dim)).permute((0, 2, 1, 3))
        k = k.view((batch_size, -1, self.n_heads, self.head_dim)).permute((0, 2, 1, 3))
        v = v.view((batch_size, -1, self.n_heads, self.head_dim)).permute((0, 2, 1, 3))

        # q = k = v = [batch_size, n_heads, seq_len, head_dim]

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch_size, n_heads, seq_len, seq_len]

        if mask is not None:
            # mask = [batch_size, 1, 1, seq_len]
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch_size, n_heads, seq_len, seq_len]

        z = torch.matmul(self.dropout(attention), v)

        # z = [batch_size, n_heads, seq_len, head_dim]

        z = z.permute(0, 2, 1, 3).contiguous()

        # z = [batch_size, seq_len, n_heads, head_dim]

        z = z.reshape((batch_size, -1, self.hid_dim))

        # z = [batch_size, seq_len, hid_dim]

        z = self.ffn(z)

        # z = [batch_size, seq_len, hid_dim]

        return z, attention


class PositionwiseFFLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim):
        super().__init__()

        self.ff1 = nn.Linear(hid_dim, pf_dim)
        self.ff2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(torch.relu(self.ff1(x)))
        return self.ff2(x)


class EncoderBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.pfn_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, device)
        self.pfn = PositionwiseFFLayer(hid_dim, pf_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, src, src_mask):
        # src = [batch_size, seq_len, hid_dim]

        z, attn = self.self_attn(src, src, src, mask=src_mask)
        z = self.self_attn_layer_norm(self.dropout(z) + src)

        # z = [batch_size, seq_len, hid_dim]

        z_ = self.pfn(z)
        z = self.pfn_layer_norm(self.dropout(z_) + z)

        return z, attn


class Encoder(nn.Module):
    def __init__(
        self, input_dim, hid_dim, n_heads, pf_dim, n_layers, device, max_length=600
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.text_embeddings = nn.Embedding(input_dim, hid_dim)
        self.pos_embeddings = nn.Embedding(max_length, hid_dim)
        self.blocks = nn.ModuleList(
            [EncoderBlock(hid_dim, n_heads, pf_dim, device) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(0.2)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch_len, src_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch_size, max_length]
        x = self.dropout(
            (self.text_embeddings(src) * self.scale) + self.pos_embeddings(pos)
        )

        # x = [batch_size, seq_len, hid_dim]

        for block in self.blocks:
            x, attn = block(x, src_mask)

        # x = [batch_size, n_layers, seq_len, hid_dim]

        return x, attn


class DecoderBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.pfn_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, device)
        self.cross_attn = MultiHeadAttentionLayer(hid_dim, n_heads, device)
        self.pfn = PositionwiseFFLayer(hid_dim, pf_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, ques, psg_enc, q_mask, p_mask):
        z, _ = self.self_attn(ques, ques, ques, q_mask)
        z = self.self_attn_layer_norm(self.dropout(z) + ques)

        z_, attn = self.cross_attn(z, psg_enc, psg_enc, p_mask)
        z = self.cross_attn_layer_norm(self.dropout(z_) + z)

        z_ = self.pfn(z)
        z = self.pfn_layer_norm(self.dropout(z_) + z)

        return z, attn


class Decoder(nn.Module):
    def __init__(
        self, output_dim, hid_dim, n_heads, pf_dim, n_layers, device, max_length=600
    ):
        super().__init__()
        self.device = device
        self.text_embeddings = nn.Embedding(output_dim, hid_dim)
        self.pos_embeddings = nn.Embedding(max_length, hid_dim)

        self.blocks = nn.ModuleList(
            [DecoderBlock(hid_dim, n_heads, pf_dim, device) for _ in range(n_layers)]
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, src_enc, trg_mask, src_mask):

        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        # pos = [batch size, ques_len]

        trg = self.dropout(
            (self.text_embeddings(trg) * self.scale) + self.pos_embeddings(pos)
        )

        # trg = [batch size, ques len, hid dim]

        for block in self.blocks:
            trg, attn = block(trg, src_enc, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attn = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attn


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        # psg = [batch size, psg len]
        # ques = [batch size, ques len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        src_enc, _ = self.encoder(src, src_mask)

        # enc_src = [batch size, psg len, hid dim]

        output, attention = self.decoder(trg, src_enc, trg_mask, src_mask)

        return output, attention
