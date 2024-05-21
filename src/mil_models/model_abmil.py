import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf
from .model_configs import ABMILConfig


class ABMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mlp = create_mlp(in_dim=config.in_dim,
                              hid_dims=[config.embed_dim] *
                              (config.n_fc_layers - 1),
                              dropout=config.dropout,
                              out_dim=config.embed_dim,
                              end_with_fc=False)

        if config.gate:
            self.attention_net = Attn_Net_Gated(L=self.config.embed_dim,
                                                D=config.attn_dim,
                                                dropout=config.dropout,
                                                n_classes=1)
        else:
            self.attention_net = Attn_Net(L=config.embed_dim,
                                          D=config.attn_dim,
                                          dropout=config.dropout,
                                          n_classes=1)

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)
        self.n_classes = config.n_classes

        self.mode = mode

    def forward_attention(self, h, attn_only=False):
        # B: batch size
        # N: number of instances per WSI
        # L: input dimension
        # K: number of attention heads (K = 1 for ABMIL)
        # h is B x N x L
        h = self.mlp(h)
        # h is B x N x D
        A = self.attention_net(h)  # B x N x K
        A = torch.transpose(A, -2, -1)  # B x K x N 
        if attn_only:
            return A
        else:
            return h, A

    def forward_no_loss(self, h, attn_mask=None):
        h, A = self.forward_attention(h)
        A_raw = A
        # A is B x K x N 
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(A, h).squeeze(dim=1) # B x K x C --> B x C
        logits = self.classifier(M)

        out = {'logits': logits, 'attn': A, 'feats': h, 'feats_agg': M}

        return out
    
    def forward(self, h, model_kwargs={}):

        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict


# class ABMILSurv(ABMIL):

#     def __init__(self, config: ABMILConfig):
#         super().__init__(config)

#     def forward(self, h, attn_mask=None, label=None, censorship=None, loss_fn=None):
#         out = self.forward_no_loss(h, attn_mask=attn_mask)
#         logits = out['logits']

#         results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)

#         return results_dict, log_dict