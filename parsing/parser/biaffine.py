# -*- coding: utf-8 -*-
# Drew plenty of inspiration from Erick Fonseca's pyturbo

from __future__ import division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import functional as F

from nonproj import nonprojective_marginals, ParserResult



class DeepBiaffineScorer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.h_mlp = nn.Linear(input_size, hidden_size)  # (Wh, bh)
        self.m_mlp = nn.Linear(input_size, hidden_size)  # (Wm, bm)
        self.V = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        torch.nn.init.normal_(self.V, 0, 0.01)

    def forward(self, Xh, Xm, lengths=None):

        h = torch.relu(self.h_mlp(Xh))
        h = self.dropout(h)
        m = torch.relu(self.m_mlp(Xm))
        m = self.dropout(m)

        scores = torch.einsum('bhu,uv,bmv->bhm', h, self.V, m)

        # to match Erick's code, we would additionally need bias terms:
        # score_ij = h_i' V m_j + a'h_i + b'm_j + c
        # Ericks weight matrix is [V, a; b, c].
        # This seems to not impact performance much when using BERT.
        # Caio says this isn't useful in dep parsing, but is useful for
        # constituency and reentrancy.

        return ParserResult(scores=scores)


class UNNBiaffineScorer(DeepBiaffineScorer):

    def __init__(self, input_size, hidden_size, dropout=0, n_iter=10):
        super().__init__(input_size, hidden_size, dropout)
        self.n_iter = n_iter

    def _update_H(self, Xh, M, Y):
        """
        Minimize energy wrt H.

        Energy expression is the sum of:

        E_HX = (H, Wh.Xh)
        E_YHM = (H, V.M'.Y')
        E_H = (H, bh.1') + Omega(H)

        where Omega is |H|_F + id(R+)
        so nabla Omega* is ReLU.
        """

        # H_ = Wh.Xh + bh.1'
        H_ = self.h_mlp(Xh)

        # tricky calculation: H' = V.M'.Y'
        # so H = YMV'
        # H  is B x N+1 x D
        # Y  is B x N+1 x N
        # M  is B x N   x D
        # V' is D x D

        # WARNING: do we need to mask anything out?
        # could be done via Y.

        YMVp = Y @ M @ self.V.T # B x N+1 x D
        return torch.relu(H_ + YMVp)

    def _update_M(self, Xm, H, Y):
        """
        Minimize energy wrt M.

        Energy expression is the sum of:

        E_MX = (M, Wm.Xm)
        E_YHM = (H, V.M'.Y')
        E_M = (M, bm.1') + Omega(M)
        """

        M_ = self.m_mlp(Xm)

        # tricky calculation: M' = V'.H'.Y
        # so M = Y'HV
        # M  is B x N   x D
        # Y  is B x N+1 x N
        # H  is B x N+1 x D
        # V is D x D

        YpHV = Y.transpose(-2, -1) @ H @ self.V  # B x N x D
        return torch.relu(M_ + YpHV)

    def _update_Y(self, H, M, lengths):
        # potentials; H @ self.V @ M.T
        scores = torch.einsum('bhu,uv,bmv->bhm', H, self.V, M)
        logZ, entr, Y = nonprojective_marginals(scores, lengths)
        return Y, logZ, entr, scores

    def _energy(self, H, H0, M, M0, logZ):
        # currently only implemented for batch size of 1
        assert H.shape[0] == 1

        energy = ( -(H * H0).sum()
                   -(M * M0).sum()
             +.5*(H ** 2).sum()
             +.5*(M ** 2).sum()
             - logZ)

        return energy.item()

    def forward(self, Xh, Xm, lengths, compute_energy=False):

        # initialize with standard forward pass.
        H0 = self.h_mlp(Xh)
        M0 = self.m_mlp(Xm)

        H_mask = self.dropout(torch.ones_like(H0))
        M_mask = self.dropout(torch.ones_like(M0))

        H = H_mask * torch.relu(H0)
        M = M_mask * torch.relu(M0)
        Y, logZ, entr, scores = self._update_Y(H, M, lengths)

        energies = []

        if compute_energy:
            energies.append(self._energy(H, H0, M, M0, logZ))

        # coordinate descent iterations on the energy
        for _ in range(1, self.n_iter):
            H_next = H_mask * self._update_H(Xh, M, Y)
            M_next = M_mask * self._update_M(Xm, H, Y)
            Y_next, logZ, entr, scores = self._update_Y(H_next, M_next, lengths)
            H, M, Y = H_next, M_next, Y_next

            if compute_energy:
                energies.append(self._energy(H, H0, M, M0, logZ))

        return ParserResult(
            scores=scores,
            lengths=lengths,
            logZ=logZ,
            mu=Y,
            entropy=entr,
            energies=energies)
