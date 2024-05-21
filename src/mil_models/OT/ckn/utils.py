# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import torch

EPS = 1e-4


def gaussian_filter_1d(size, sigma=None):
    """Create 1D Gaussian filter
    """
    if size == 1:
        return torch.ones(1)
    if sigma is None:
        sigma = (size - 1.)/(2.*math.sqrt(2))
    m = float((size - 1) // 2)
    filt = torch.arange(-m, m+1)
    filt = torch.exp(-filt.pow(2)/(2.*sigma*sigma))
    return filt/torch.sum(filt)


def init_kmeans(x, n_clusters, n_local_trials=None, use_cuda=False):
    n_samples, n_features = x.size()
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters.cuda()

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    clusters[0] = x[np.random.randint(n_samples)]

    closest_dist_sq = 1 - clusters[[0]].mm(x.t())
    closest_dist_sq = closest_dist_sq.view(-1)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1), rand_vals)
        distance_to_candidates = 1 - x[candidate_ids].mm(x.t())

        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = torch.min(closest_dist_sq,
                                    distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return clusters


def spherical_kmeans(x, n_clusters, max_iters=100, verbose=True,
                     init=None, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    n_samples, n_features = x.size()
    if init == "kmeans++":
        print(init)
        clusters = init_kmeans(x, n_clusters, use_cuda=use_cuda)
    else:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf

    for n_iter in range(max_iters):
        # assign data points to clusters
        cos_sim = x.mm(clusters.t())
        tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                # clusters[j] = x[random.randrange(n_samples)]
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm()

        if torch.abs(prev_sim - sim)/(torch.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters


def normalize_(x, p=2, dim=-1):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=EPS))
    return x


def flip(x, dim=-1):
    """Reverse a tensor along given axis
    can be removed later when Pytorch updated
    """
    reverse_indices = torch.arange(x.size(dim) - 1, -1, -1)
    reverse_indices = reverse_indices.type_as(x.data).long()
    return x.index_select(dim=dim, index=reverse_indices)


def proj_on_simplex(x, axis=0, r=1., inplace=True):
    d = x.size(axis)
    mu, indices = torch.sort(x, dim=axis, descending=True)
    diag = torch.cumsum(mu, dim=axis) - r
    theta = diag / torch.arange(1., d+1).view(-1, 1).expand_as(diag)
    indices = torch.sum((mu > theta).long(), dim=axis, keepdim=True) - 1
    theta = torch.gather(theta, dim=axis, index=indices)
    if inplace:
        x.add_(-theta).clamp_(min=0)
        return x
    return torch.clamp(x - theta, min=0)
