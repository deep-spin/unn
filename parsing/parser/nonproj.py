"""non-projective inference stuff"""

import numpy as np

from typing import Optional
from dataclasses import dataclass

import torch

from logdecomp.functions import logdet_and_inv_exp
from lpsmap import TorchFactorGraph, DepTree
from torch_struct.deptree import DepTree as TSDepTree
from logdecomp.functions import logdet_and_inv_exp

NINF = float('-inf')


# packing and unpacking
#######################

def pack(scores):
    """map a batched score tensor [B, N+1, N] into a packed one [B, N, N].
    The first row of each batch replaces the diagonal

    Warning: makes changes in place.
    """

    root_scores = scores[:, 0, :]
    scores = scores[:, 1:, :]
    torch.diagonal(scores, dim1=-2, dim2=-1)[...] = root_scores
    return scores


def unpack(scores):
    """map a batched score tensor [B, N, N] into an unpacked one [B, N+1, N].
    The diagonal moves to the first row of each batch.

    Warning: makes changes in place.
    """

    diag_ref = torch.diagonal(scores, dim1=-2, dim2=-1)
    root = diag_ref.clone()
    diag_ref.fill_(0)
    return torch.cat((root.unsqueeze(-2), scores), dim=-2)


# argmax using lp-sparsemap package
###################################

def argmax_nonproj(arc_scores):
    fg = TorchFactorGraph()
    s = arc_scores.contiguous()
    u = fg.variable_from(s)
    fg.add(DepTree(u, packed=True, projective=False))
    fg.solve_map(autodetect_acyclic=True)
    return u.value


def argmax_batched(scores_batched, lengths):
    # scores_batched is [B, N, N]
    # transpose because LP-SparseMAP factor differs from torch_struct
    S = scores_batched.cpu().transpose(-2, -1)
    n = S.shape[0]
    A = [argmax_nonproj(S[i, :lengths[i], :lengths[i]]).T for i in range(n)]

    return A


class ParserAccuracy:
    def __init__(self):
        self.wrong = 0
        self.total = 0
        self.allarcs_correct = 0
        self.allarcs_total = 0
        self.full_correct = 0
        self.full_total = 0

    def update(self, wrong, total, allarcs_correct, allarcs_total, full_correct, full_total):
        self.wrong += wrong
        self.total += total
        self.allarcs_correct += allarcs_correct
        self.allarcs_total += allarcs_total
        self.full_correct += full_correct
        self.full_total += full_total

    def value(self):
        return 1 - self.wrong / self.total, \
                self.allarcs_correct/self.allarcs_total, \
                self.full_correct/self.full_total


def factored_log_prob(scores, flat_labels, lengths):
    gold_parts = (TSDepTree.to_parts(flat_labels, lengths=lengths)
                  .to(dtype=torch.bool).detach())

    log_prob = torch.log_softmax(scores, dim=1)
    return log_prob[gold_parts].sum()


# log-likelihood and marginals using the logdecomp package
###################################


def logz_and_marginals_mtt_torchstruct(arc_scores, lengths, eps=0):
    if lengths is not None:
        batch, N, N = arc_scores.shape
        x = torch.arange(N, device=arc_scores.device).expand(batch, N)
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=arc_scores.device)
        lengths = lengths.unsqueeze(1)
        x = x < lengths
        det_offset = torch.diag_embed((~x).float())
        x = x.unsqueeze(2).expand(-1, -1, N)
        mask = torch.transpose(x, 1, 2) * x
        mask = mask.float()
        mask[mask == 0] = float("-inf")
        mask[mask == 1] = 0
        arc_scores = arc_scores + mask

    input = arc_scores
    eye = torch.eye(input.shape[1], device=input.device)
    laplacian = input.exp() + eps
    lap = laplacian.masked_fill(eye != 0, 0)
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)

    if lengths is not None:
        lap += det_offset

    lap[:, 0] = torch.diagonal(input, 0, -2, -1).exp()

    logZ = lap.logdet()

    inv_laplacian = lap.inverse()
    factor = (
        torch.diagonal(inv_laplacian, 0, -2, -1)
        .unsqueeze(2)
        .expand_as(input)
        .transpose(1, 2)
    )
    term1 = input.exp().mul(factor).clone()
    term2 = input.exp().mul(inv_laplacian.transpose(1, 2)).clone()
    term1[:, :, 0] = 0
    term2[:, 0] = 0
    output = term1 - term2
    roots_output = (
        torch.diagonal(input, 0, -2, -1)
        .exp()
        .mul(inv_laplacian.transpose(1, 2)[:, 0])
    )

    output = output + torch.diag_embed(roots_output, 0, -2, -1)
    return logZ, output


def logz_and_marginals_mtt(scores_packed, lengths):
    """Non-projective dependency parsing log Z and marginals.

    Uses the matrix-tree theorem.

    Based on torch-struct code,
    replacing logdet/inv with a single call to logdecomp
    """

    dtype = scores_packed.dtype
    X = scores_packed.to(dtype=torch.float64)

    batch, N, N = X.shape

    # build mask tensor for lengths. We almost don't use this at all.
    ix = torch.arange(N, device=X.device).expand(batch, N)
    lengths_t = (torch.as_tensor(lengths, device=X.device)
                      .unsqueeze(1))
    ix = ix < lengths_t
    # det_offset = torch.diag_embed((~ix).to(dtype=X.dtype))
    ix = ix.unsqueeze(2).expand(-1, -1, N)
    mask = torch.transpose(ix, 1, 2) * ix

    # if we were to use it, it would be like this
    # mask = mask.float()
    # mask[mask == 0] = -9999
    # mask[mask == 1] = 0
    # X = X + mask

    # this gives nan if masking with NINF, ok if mask -9999
    # log_lap = X.clone().masked_fill(eye != 0, NINF)
    # log_diag = torch.logsumexp(log_lap, dim=1)
    # log_lap = log_lap.masked_fill(eye != 0, 0) + torch.diag_embed(log_diag)

    # replacing with
    log_lap = X.clone()
    eye = torch.eye(N, device=X.device, dtype=bool)
    X_tmp = X.clone().masked_fill(eye, NINF)

    for k in range(batch):
        Xk = X_tmp[k, :lengths[k], :lengths[k]]
        diag = torch.logsumexp(Xk, dim=0)
        log_lap[k, :lengths[k], :lengths[k]].diagonal().copy_(diag)

    # set root scores to first row
    log_lap[:, 0] = torch.diagonal(X, 0, -2, -1)

    # sign bit: which coordinates of the laplacian have negative sign?
    sign = eye.clone()
    sign[0, :] = 1
    sign = ~sign

    logdet, inv = logdet_and_inv_exp(log_lap.cpu(), lengths.cpu(), sign.cpu())
    logdet = logdet.to(device=sign.device, dtype=log_lap.dtype)
    inv = inv.to(device=sign.device, dtype=log_lap.dtype)

    factor = (
        torch.diagonal(inv, 0, -2, -1)
        .unsqueeze(2)
        .expand_as(X)
        .transpose(1, 2)
    )

    # term1 needs to be zeroed out
    expX = X.clone().exp()
    term1 = expX.masked_fill(~mask, 0).mul(factor)
    term2 = expX.mul(inv.transpose(1, 2))
    term1[:, :, 0] = 0
    term2[:, 0] = 0

    mu = term1 - term2
    mu_root = (
        torch.diagonal(X, 0, -2, -1)
        .exp()
        .mul(inv.transpose(1, 2)[:, 0])
    )
    mu = mu + torch.diag_embed(mu_root, 0, -2, -1)

    mu = mu.to(dtype=dtype)
    logdet = logdet.to(dtype=dtype)
    return logdet, mu


def nonprojective_marginals(scores, lengths):
    mask_one = lengths.le(1)
    mask_long = lengths.gt(1)

    # print(lengths)

    scores_len_one = scores[mask_one,:,:] 
    scores_len_long = scores[mask_long,:,:]

    if scores_len_long.shape[0] > 0:
        scores_packed = pack(scores_len_long.clone())

        # call matrix-tree theorem
        logZ, mu_packed = logz_and_marginals_mtt(scores_packed, lengths[mask_long])
        mu = unpack(mu_packed.clone())

        # compute entropy (generally optional, we need it for CD)
        expected_score = (mu_packed * scores_packed).sum(dim=-2).sum(dim=-1)
        entr = logZ - expected_score
        entr = torch.clip(entr, min=0)  # numerical issues

    if scores_len_one.shape[0] > 0:
        logZ_ones = scores_len_one[:,0,0]
        entr_ones = torch.zeros(scores_len_one.shape[0], device=scores.device)
        mu_ones = torch.zeros_like(scores_len_one, device=scores.device)
        mu_ones[:,0,0] = 1

        # Make tensors for storing all lengths
        logZ_full = torch.ones(scores.shape[0], device=scores.device)
        entr_full = torch.ones(scores.shape[0], device=scores.device)
        mu_full = torch.ones_like(scores, device=scores.device)

        logZ_full[mask_one] = logZ_ones
        entr_full[mask_one] = entr_ones
        mu_full[mask_one] = mu_ones

        if scores_len_long.shape[0] > 0:
            logZ_full[mask_long] = logZ
            entr_full[mask_long] = entr
            mu_full[mask_long] = mu

        logZ = logZ_full
        entr = entr_full
        mu = mu_full

    return logZ, entr, mu


@dataclass
class ParserResult:
    """The result of running the forward pass in a dependency parser.

    scores and mu are UNPACKED -- caution.

    """
    scores: torch.Tensor
    lengths: torch.Tensor
    logZ: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    energies: Optional[torch.Tensor] = None

    def log_prob(self, y_true):
        """Compute the log probability of the true labels"""

        batch, N, N = self.scores.shape

        if self.logZ is None:
            (self.logZ,
             self.entropy,
             self.mu) = nonprojective_marginals(self.scores, self.lengths)

        # go from heads vector to 0-1 indicator matrix
        # careful, we don't trust this code. It hides bugs.
        # We had a bug here because of inclusion of 2-4 tokens in UD.
        scores_packed = pack(self.scores.clone())
        Y_true = (TSDepTree.to_parts(y_true, lengths=self.lengths)
                           .to(device=scores_packed.device))

        score_true = (Y_true * scores_packed).sum(dim=-2).sum(dim=-1)

        log_prob = score_true - self.logZ
        # assert is too harsh, numeric issues possible.
        # assert log_prob.max().item() <= 0
        max_log_prob = log_prob.max().item()
        if max_log_prob > 0:
            print("Warning: some log_prob is above 0. Clipping.", max_log_prob)
        log_prob = torch.clamp(log_prob, max=0)
        return log_prob

    def accuracy_terms(self, y_true):

        # Get MAP from lp-sparsemap
        scores_packed = pack(self.scores.clone())
        pred_parts = argmax_batched(scores_packed, self.lengths)
        gold_parts = TSDepTree.to_parts(y_true, self.lengths).cpu()

        # Connting wrong and total arcs
        wrong = 0
        total = 0

        # Counting correct full sentences
        full_correct = 0
        full_total = 0

        # Counting correct all arcs per head
        all_arcs_correct = 0
        all_arcs_total = 0

        # print('pred_parts', len(pred_parts), pred_parts)
        # print('y_true', len(y_true), y_true)
        for b in range(len(pred_parts)):
            pred_ = pred_parts[b]
            gold_ = gold_parts[b, :self.lengths[b], :self.lengths[b]].to(pred_.dtype)

            # print('pred_', len(pred_), pred_)
            # print('gold_', len(gold_), gold_)

            # Counting arcs
            wrong += ((pred_ - gold_).abs().sum() / 2).item()
            total += gold_.sum().item()

            # Counting all arcs per head
            head_ids = gold_.sum(dim=1).nonzero().flatten() # Get the head indices
            # When we sum the pred and gold, the matching arcs will have a score of 2
            sh = (pred_[head_ids,:]+gold_[head_ids,:]).sum(dim=1) # summed heads
            dh = (2*gold_[head_ids,:]).sum(dim=1) # doubled heads
            # The number of heads with all arcs matched should have equal sums in the rows with matched 2s
            all_arcs_correct += (sh==dh).sum() 
            all_arcs_total += len(head_ids) # Add the number of heads to the total of calculating arcs

            # Counting full sentences
            if torch.allclose(pred_, gold_):
                # print('allclose')
                full_correct += 1
            full_total += 1

        return wrong, total, all_arcs_correct, all_arcs_total, full_correct, full_total
