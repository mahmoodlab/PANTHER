"""
Optimal transport (OT)-based aggregation

Ref:
    Mialon, Grégoire, et al. "A trainable optimal transport embedding for feature aggregation and its relationship to attention." arXiv preprint arXiv:2006.12065 (2020).
    https://github.com/claying/OTK
"""

from .components import create_mlp, predict_surv, predict_clf, predict_emb
from .OT.otk.layers import OTKernel
from utils.proto_utils import check_prototypes
from utils.file_utils import save_pkl, load_pkl

import torch
from torch import nn
import numpy as np
import pdb

class OT(nn.Module):
    """
    OTK method without bells and whistles (no convolutional kernel & no bioembedding)
    """

    def __init__(self, config, mode):
        super().__init__()
        self.config = config

        self.attention = OTKernel(in_dim=config.in_dim, 
                                  out_size=config.out_size, 
                                  distance=config.distance,
                                  heads=config.heads, 
                                  max_iter=config.max_iter, 
                                  eps=config.ot_eps)

        self.out_type = config.out_type
        self.mode = mode

        if self.out_type == 'allcat':
            self.out_features = config.in_dim * config.out_size * config.heads
        elif self.out_type == 'weight_avg_mean':
            self.out_features = config.in_dim * config.heads
        else:
            raise NotImplementedError(f"OT Not implemented for {self.out_type}!")

        self.nclass = config.n_classes

        check_prototypes(config.out_size, config.in_dim, config.load_proto, config.proto_path)

        if config.load_proto:
            if config.proto_path.endswith('pkl'):
                weights = load_pkl(config.proto_path)['prototypes'].squeeze()
            elif config.proto_path.endswith('npy'):
                weights = np.load(config.proto_path)
            weights = torch.from_numpy(weights)
            self.attention.weight.data.copy_(weights)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        B = x.shape[0]
        # out = self.attention(x.permute(0, 2, 1))    # (batch_size, out_size, in_dim)
        out = self.attention(x)    # (batch_size, out_size, in_dim)

        if self.out_type == 'allcat':
            out = out.reshape(B, -1)
        elif self.out_type == 'weight_avg_mean':
            out = torch.mean(out, dim=1)
        else:
            raise NotImplementedError(f"OTK Not implemented for {self.out_type}!")
        
        return {'repr': out}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y