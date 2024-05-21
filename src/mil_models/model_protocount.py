import torch
import torch.nn as nn
import os
import numpy as np
import pdb

from .components import predict_surv, predict_clf, predict_emb
from utils.file_utils import save_pkl, load_pkl

class ProtoCount(nn.Module):
    """
    WSI is represented as a prototype-count vector
    """
    def __init__(self, config, mode):
        super().__init__()

        assert config.load_proto, "Prototypes must be loaded!"
        assert os.path.exists(config.proto_path), "Path {} doesn't exist!".format(config.proto_path)

        self.config = config
        proto_path = config.proto_path

        if proto_path.endswith('pkl'):
            weights = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            weights = np.load(proto_path)

        self.n_proto = config.out_size
        self.prototypes = torch.from_numpy(weights).float()
        self.mode = mode

        emb_dim = config.in_dim

    def representation(self, x):
        """
        Compute the distance Eulcidean between prototypes and the patch features

        Args:
            x:

        Returns:

        """
        self.prototypes = self.prototypes.to(x.device)
        dist = torch.cdist(self.prototypes, x, p=2) # (1 x n_proto x n_instances)
        c_identity = torch.argmin(dist.squeeze(), dim=0) # (n_instances)
        counts = torch.bincount(c_identity, minlength=self.n_proto).unsqueeze(dim=0).float() # (1, n_proto)
        counts = counts / torch.norm(counts, dim=1)

        return {'repr': counts}

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