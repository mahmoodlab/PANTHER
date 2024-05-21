# Model initiation for PANTHER

from torch import nn
import numpy as np

from .components import predict_surv, predict_clf, predict_emb
from .PANTHER.layers import PANTHERBase
from utils.proto_utils import check_prototypes


class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, config, mode):
        super(PANTHER, self).__init__()

        self.config = config
        emb_dim = config.in_dim

        self.emb_dim = emb_dim
        self.heads = config.heads
        self.outsize = config.out_size
        self.load_proto = config.load_proto
        self.mode = mode

        check_prototypes(config.out_size, self.emb_dim, self.load_proto, config.proto_path)
        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, p=config.out_size, L=config.em_iter,
                         tau=config.tau, out=config.out_type, ot_eps=config.ot_eps,
                         load_proto=config.load_proto, proto_path=config.proto_path,
                         fix_proto=config.fix_proto)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out, qqs = self.panther(x)
        return {'repr': out, 'qq': qqs}

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