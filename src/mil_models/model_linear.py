import torch
import torch.nn as nn
import pdb

from .components import create_mlp, create_mlp_with_dropout, process_surv, process_clf

class LinearEmb(nn.Module):
    """
    Linear fully-connected layer from slide representation to output
    """
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.in_dim, config.n_classes, bias=False)
        self.n_classes = config.n_classes
        self.mode = mode

    def forward_no_loss(self, h, attn_mask=None):
        logits = self.classifier(h)
        out = {'logits': logits}
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


#
# MLP per prototype
#
class IndivMLPEmb(nn.Module):
    """
    Comprised of three MLP (in sequence), each of which can be enabled/disabled and configured accordingly
    - Shared: Shared MLP across prototypes for feature dimension reduction
    - Indiv: Individual MLP per prototype
    - Post: Shared MLP across prototypes for final feature dimension reduction
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_classes = config.n_classes
        self.p = config.p
        mlp_func = create_mlp_with_dropout

        if config.shared_mlp:
            self.shared_mlp = mlp_func(in_dim=config.in_dim,
                                    hid_dims=[config.shared_embed_dim] *
                                            (config.n_fc_layers - 1),
                                    dropout=config.shared_dropout,
                                    out_dim=config.shared_embed_dim,
                                    end_with_fc=False)
            next_in_dim = config.shared_embed_dim
        else:
            self.shared_mlp = nn.Identity()
            next_in_dim = config.in_dim
            
        if config.indiv_mlps:
            self.indiv_mlps = nn.ModuleList([mlp_func(in_dim=next_in_dim,
                                hid_dims=[config.indiv_embed_dim] *
                                        (config.n_fc_layers - 1),
                                dropout=config.indiv_dropout,
                                out_dim=config.indiv_embed_dim,
                                end_with_fc=False) for i in range(config.p)])
            next_in_dim = config.p * config.indiv_embed_dim
        else:
            self.indiv_mlps = nn.ModuleList([nn.Identity() for i in range (config.p)])
            next_in_dim = config.p * next_in_dim

        if config.postcat_mlp:
            self.postcat_mlp = mlp_func(in_dim=next_in_dim,
                                    hid_dims=[config.postcat_embed_dim] *
                                            (config.n_fc_layers - 1),
                                    dropout=config.postcat_dropout,
                                    out_dim=config.postcat_embed_dim,
                                    end_with_fc=False)
            next_in_dim = config.postcat_embed_dim
        else:
            self.postcat_mlp = nn.Identity()
        
        self.classifier = nn.Linear(next_in_dim,
                                    config.n_classes,
                                    bias=False)

    def forward_no_loss(self, h, attn_mask=None):
        h = self.shared_mlp(h)
        h = torch.stack([self.indiv_mlps[idx](h[:, idx, :]) for idx in range(self.p)], dim=1)
        h = h.reshape(h.shape[0], -1)   # (n_samples, n_proto * config.indiv_embed_dim)
        h = self.postcat_mlp(h)
        logits = self.classifier(h)
        out = {'logits': logits}
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