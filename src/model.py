# -*- coding: utf8 -*-
#
import torch
from torch import nn

from src.algo import cky
from src.layers.affine import Biaffine
from src.layers.mlp import MLP
from src.layers.transformer import TransformerEmbedding
from src.layers.treecrf import CRFConstituency
from src.transform import get_tags
from src.utils import logger


class CRFConstituencyModel(nn.Module):
    def __init__(self, transformer, n_labels, n_span_mlp=500, n_label_mlp=100, mlp_dropout=0.33):
        super(CRFConstituencyModel, self).__init__()

        self.encoder = TransformerEmbedding(model=transformer,
                                            n_layers=4,
                                            pooling='mean',
                                            pad_index=0,
                                            dropout=0.33,
                                            requires_grad=True)
        self.tags_embedding = nn.Embedding(len(get_tags()), embedding_dim=64)
        self.span_mlp_l = MLP(n_in=self.encoder.n_out + self.tags_embedding.embedding_dim, n_out=n_span_mlp, dropout=mlp_dropout)
        self.span_mlp_r = MLP(n_in=self.encoder.n_out + self.tags_embedding.embedding_dim, n_out=n_span_mlp, dropout=mlp_dropout)
        self.label_mlp_l = MLP(n_in=self.encoder.n_out + self.tags_embedding.embedding_dim, n_out=n_label_mlp, dropout=mlp_dropout)
        self.label_mlp_r = MLP(n_in=self.encoder.n_out + self.tags_embedding.embedding_dim, n_out=n_label_mlp, dropout=mlp_dropout)

        self.span_attn = Biaffine(n_in=n_span_mlp, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.crf = CRFConstituency()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, tags=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.
            tags
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible constituents.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each constituent.
        """

        x = self.encoder(words)

        # x_f, x_b = x.chunk(2, -1)
        # x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        tags_x = self.tags_embedding(tags)
        x = torch.cat([x, tags_x], dim=-1)

        span_l = self.span_mlp_l(x)
        span_r = self.span_mlp_r(x)
        label_l = self.label_mlp_l(x)
        label_r = self.label_mlp_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label

    def loss(self, s_span, s_label, charts, mask, mbr=True):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each constituent.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels. Positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and original constituent scores
                of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.crf(s_span, mask, span_mask, mbr)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss
        if span_loss.item() < 0:
            logger.warning(f'span_loss litter than 0, current value: {span_loss.item()}')
        if label_loss.item() < 0:
            logger.warning(f'label_loss litter than 0, current value: {span_loss.item()}')
        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each constituent.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            list[list[tuple]]:
                Sequences of factorized labeled trees traversed in pre-order.
        """

        span_preds = cky(s_span.unsqueeze(-1), mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j, _ in cons] for cons, labels in zip(span_preds, label_preds)]
