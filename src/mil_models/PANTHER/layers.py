#
# Codebase adapted from Kim, M. "Differentiable Expectation-Maximization for Set Representation Learning ", ICLR, 2022
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from .networks import DirNIWNet
import pdb

class PANTHERBase(nn.Module):
    """
    Args:
    - p (int): Number of prototypes
    - d (int): Feature dimension
    - L (int): Number of EM iterations
    - out (str): Ways to merge features
    - ot_eps (float): eps
    """
    def __init__(self, d, p=5, L=3, tau=10.0, out='allcat', ot_eps=0.1,
                 load_proto=True, proto_path='.', fix_proto=True):
        super(PANTHERBase, self).__init__()

        self.L = L
        self.tau = tau
        self.out = out

        self.priors = DirNIWNet(p, d, ot_eps, load_proto, proto_path, fix_proto)

        if out == 'allcat':  # Concatenates pi, mu, cov
            self.outdim = p + 2*p*d
        elif out == 'weight_param_cat': # Concatenates mu and cov weighted by pi
            self.outdim = 2 * p * d
        elif 'select_top' in out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            assert numOfproto <= p
            self.outdim = numOfproto * 2 * d + numOfproto
        elif 'select_bot' in out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            assert numOfproto <= p
            self.outdim = numOfproto * 2 * d + numOfproto
        elif out == 'weight_avg_all':
            self.outdim = 2 * d
        elif out == 'weight_avg_mean':
            self.outdim = d
        else:
            raise NotImplementedError("Out mode {} not implemented".format(out))

    def forward(self, S, mask=None):
        """
        Args
        - S: data
        """
        B, N_max, d = S.shape
        
        if mask is None:
            mask = torch.ones(B, N_max).to(S)
        
        pis, mus, Sigmas, qqs = [], [], [], []
        pi, mu, Sigma, qq = self.priors.map_em(S, 
                                                    mask=mask, 
                                                    num_iters=self.L, 
                                                    tau=self.tau, 
                                                    prior=self.priors())

        pis.append(pi)
        mus.append(mu)
        Sigmas.append(Sigma)
        qqs.append(qq)

        pis = torch.stack(pis, dim=2) # pis: (n_batch x n_proto x n_head)
        mus = torch.stack(mus, dim=3) # mus: (n_batch x n_proto x instance_dim x n_head)
        Sigmas = torch.stack(Sigmas, dim=3) # Sigmas: (n_batch x n_proto x instance_dim x n_head)
        qqs = torch.stack(qqs, dim=3)

        if self.out == 'allcat':
            out = torch.cat([pis.reshape(B,-1),
                mus.reshape(B,-1), Sigmas.reshape(B,-1)], dim=1)
        elif self.out == 'weight_param_cat':
            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h].reshape(B, -1), mus[..., h], Sigmas[..., h]
                mu_weighted = pi[..., None] * mu  # (n_batch, n_proto, instance_dim)
                Sigma_weighted = pi[..., None] * Sigma  # (n_batch, n_proto, instance_dim)

                out.append(mu_weighted.reshape(B, -1))
                out.append(Sigma_weighted.reshape(B, -1))

            out = torch.cat(out, dim=1)

        elif self.out == 'weight_avg_all':
            """
            Take weighted average of mu and sigmas according to estimated pi
            """
            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h].reshape(B, 1, -1), mus[..., h], Sigmas[..., h]
                mu_weighted = torch.bmm(pi, mu).squeeze(dim=1)  # (n_batch, instance_dim)
                Sigma_weighted = torch.bmm(pi, Sigma).squeeze(dim=1)  # (n_batch, instance_dim)

                out.append(mu_weighted)
                out.append(Sigma_weighted)

            out = torch.cat(out, dim=1)

        elif self.out == 'weight_avg_mean':
            """
            Take weighted average of mu according to estimated pi
            """
            out = []
            for h in range(self.H):
                pi, mu = pis[..., h].reshape(B, 1, -1), mus[..., h]
                mu_weighted = torch.bmm(pi, mu).squeeze(dim=1)  # (n_batch, instance_dim)

                out.append(mu_weighted)

            out = torch.cat(out, dim=1)

        elif 'select_top' in self.out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h], mus[..., h], Sigmas[..., h]
                _, indices = torch.topk(pi, numOfproto, dim=1)

                out.append(pi[:, indices].reshape(pi.shape[0], -1))
                out.append(mu[:, indices].reshape(mu.shape[0], -1))
                out.append(Sigma[:, indices].reshape(Sigma.shape[0], -1))
            out = torch.cat(out, dim=1)

        elif 'select_bot' in self.out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h], mus[..., h], Sigmas[..., h]
                _, indices = torch.topk(-pi, numOfproto, dim=1)

                out.append(pi[:, indices].reshape(pi.shape[0], -1))
                out.append(mu[:, indices].reshape(mu.shape[0], -1))
                out.append(Sigma[:, indices].reshape(Sigma.shape[0], -1))
            out = torch.cat(out, dim=1)

        else:
            raise NotImplementedError

        return out, qqs
