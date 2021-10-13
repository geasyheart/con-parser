# -*- coding: utf8 -*-
#
import torch
from torch import nn, autograd

from src.algo import stripe


class CRFConstituency(nn.Module):
    r"""
    TreeCRF for calculating partitions and marginals of constituency trees in :math:`O(n^3)` :cite:`zhang-etal-2020-fast`.
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible constituents.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid parsing over padding tokens.
                For each square matrix in a batch, the positions except upper triangular part should be masked out.
            target (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard constituents. ``True`` if a constituent exists. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = scores
        if mbr:
            marginals, = autograd.grad(logZ, scores, retain_graph=training)
        if target is None:
            return marginals
        loss = (logZ - scores[mask & target].sum()) / mask[:, 0].sum()

        return loss, marginals

    def inside(self, scores, mask):
        lens = mask[:, 0].sum(-1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
        s = torch.full_like(scores, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of constituents to iterate,
            # from constituent (0, w) to constituent (n, n+w) given width w
            n = seq_len - w

            if w == 1:
                s.diagonal(w).copy_(scores.diagonal(w))
                continue
            # [n, w, batch_size]
            s_s = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
            # [batch_size, n, w]
            s_s = s_s.permute(2, 0, 1)
            if s_s.requires_grad:
                s_s.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_s = s_s.logsumexp(-1)
            s.diagonal(w).copy_(s_s + scores.diagonal(w))

        return s[0].gather(0, lens.unsqueeze(0)).sum()

