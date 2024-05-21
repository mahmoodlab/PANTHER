import torch.nn as nn
import torch
import numpy as np
import pdb
from itertools import combinations

# Optimal Transport loss
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)

# Survival loss
class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, logits, times, censorships):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(logits=logits, y=times, c=censorships,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(logits, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    logits: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(logits).
    y: (n_batches, )
        The true time bin index label.
    c: (n_batches, )
        The censoring status indicator.
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.long()
    c = c.long()

    hazards = torch.sigmoid(logits)

    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
        censored_loss = censored_loss.mean()
        uncensored_loss = uncensored_loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
        censored_loss = censored_loss.sum()
        uncensored_loss = uncensored_loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return {'loss': loss, 'uncensored_loss': uncensored_loss, 'censored_loss': censored_loss}

def partial_ll_loss(lrisks, survival_times, event_indicators):
    """
    lrisks: log risks, B x 1
    survival_times: time bin, B x 1
    event_indicators: event indicator, B x 1
    """    
    num_uncensored = torch.sum(event_indicators, 0)
    if num_uncensored.item() == 0:
        return {'loss': torch.sum(lrisks) * 0}
    
    survival_times = survival_times.squeeze(1)
    event_indicators = event_indicators.squeeze(1)
    lrisks = lrisks.squeeze(1)

    sindex = torch.argsort(-survival_times)
    survival_times = survival_times[sindex]
    event_indicators = event_indicators[sindex]
    lrisks = lrisks[sindex]

    log_risk_stable = torch.logcumsumexp(lrisks, 0)

    likelihood = lrisks - log_risk_stable
    uncensored_likelihood = likelihood * event_indicators
    logL = -torch.sum(uncensored_likelihood)
    # negative average log-likelihood
    return {'loss': logL / num_uncensored}


class CoxLoss(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()

    def __call__(self, logits, times, censorships):
        return partial_ll_loss(lrisks = logits, survival_times=times, event_indicators=(1-censorships).float())


class SurvRankingLoss(nn.Module):
    """
    Implements the surivival ranking loss which approximates the negaive c-index; see Section 3.2 of (Luck et al, 2018) -- but be careful of the typo in their c-index formula.

    The c-index for risk scores z_1, ..., z_n is given by

    c_index = sum_{(a, b) are comparable} 1(z_a > z_b)

    where (a, b) are comparable if and only if a's event is observed and a has a strictly lower survival time than b. This ignores ties.

    We replace the indicator with a continous approximation

    1(z_a - z_b > 0 ) ~= phi(z_a - z_b)

    e.g. where phi(r) is a Relu or sigmoid function.

    The loss function we want to minimize is then

    - sum_{(a, b) are comparable} phi(z_a - z_b)

    where z_a, z_b are the risk scores output by the network.

    Parameters
    ----------
    phi: str
        Which indicator approximation to use. Must be one of ['relu', 'sigmoid'].

    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']

    References
    ----------
    Luck, M., Sylvain, T., Cohen, J.P., Cardinal, H., Lodi, A. and Bengio, Y., 2018. Learning to rank for censored survival data. arXiv preprint arXiv:1806.01984.
    """

    def __init__(self, phi='sigmoid', reduction='mean'):
        super().__init__()

        assert phi in ['sigmoid', 'relu']
        assert reduction in ['mean', 'sum']
        self.phi = phi
        self.reduction = reduction

    def forward(self, z, times, censorships):
        """
        Parameters
        ----------
        z: (batch_size, 1)
            The predicted risk scores.

        c_t: (batch_size, 2)
            first element: censorship
            second element: survival time
        """
        batch_size = z.shape[0]
        if batch_size == 1:
            # raise NotImplementedError("Batch size must be at least 2")
            return {'loss': torch.tensor(-1e5)}

        # censorship, times = c_t[:, 0], c_t[:, 1]
        events = 1 - censorships

        ##############################
        # determine comparable pairs #
        ##############################
        Z_more_risky = []
        Z_less_risky = []
        for (idx_a, idx_b) in combinations(range(batch_size), 2):
            time_a, event_a = times[idx_a], events[idx_a]
            time_b, event_b = times[idx_b], events[idx_b]

            if time_a < time_b and event_a:
                # a and b are comparable, a is more risky
                Z_more_risky.append(z[idx_a])
                Z_less_risky.append(z[idx_b])

            elif time_b < time_a and event_b:
                # a and b are comparable, b is more risky
                Z_more_risky.append(z[idx_b])
                Z_less_risky.append(z[idx_a])

        # if there are no comparable pairs then just return zero
        if len(Z_less_risky) == 0:
            # TODO: perhaps return None?
            return {'loss': None}

        Z_more_risky = torch.stack(Z_more_risky)
        Z_less_risky = torch.stack(Z_less_risky)

        # compute approximate c indices
        r = Z_more_risky - Z_less_risky
        if self.phi == 'sigmoid':
            approx_c_indices = torch.sigmoid(r)

        elif self.phi == 'relu':
            approx_c_indices = torch.relu(r)

        # negative mean/sum of c-indices
        if self.reduction == 'mean':
            return {'loss': - approx_c_indices.mean()}
        if self.reduction == 'sum':
            return {'loss': -approx_c_indices.sum()}