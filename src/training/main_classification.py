"""
Main entry point for classification downstream tasks
"""

from __future__ import print_function

import argparse
import pdb
import os
from os.path import join as j_
import sys

# internal imports
from utils.file_utils import save_pkl
from utils.utils import (seed_torch, array2list, merge_dict, read_splits, 
                         parse_model_name, get_current_time,
                         extract_patching_info)

from .trainer import train
from wsi_datasets import WSIClassificationDataset
from data_factory import tasks, label_dicts

import torch
from torch.utils.data import DataLoader, sampler

import pandas as pd
import numpy as np
import json

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']

def build_sampler(dataset, sampler_type=None):
    data_sampler = None
    if sampler_type is None:
        return data_sampler
    
    assert sampler_type in ['weighted', 'random', 'sequential']
    if sampler_type == 'weighted':
        labels = dataset.get_labels(np.arange(len(dataset)), apply_transform=True)
        uniques, counts = np.unique(labels, return_counts=True)
        weights = {uniques[i]: 1. / counts[i] for i in range(len(uniques))}
        samples_weight = np.array([weights[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        data_sampler = sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    elif sampler_type == 'random':
        data_sampler = sampler.RandomSampler(dataset)
    elif sampler_type == 'sequential':
        data_sampler = sampler.SequentialSampler(dataset)

    return data_sampler

def build_datasets(csv_splits, model_type, batch_size=1, num_workers=2,
                   train_kwargs={}, val_kwargs={}, sampler_types={'train': 'random',
                                                                  'val': 'sequential',
                                                                  'test': 'sequential'}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        print("\nSPLIT: ", k)
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        if k == 'test_nlst':
            dataset_kwargs['sample_col'] = 'case_id'
        dataset = WSIClassificationDataset(df, **dataset_kwargs)
        data_sampler = build_sampler(dataset, sampler_type=sampler_types.get(k, 'sequential'))

        # If prototype methods, each WSI will have same feature bag dimension and is batchable
        # Otherwise, we need to use batch size of 1 to accommodate to different bag size for each WSI.
        # Alternatively, we can sample same number of patch features per WSI to have larger batch.
        if model_type not in PROTO_MODELS:
            batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size

    sampler_types = {'train': args.train_sampler if args.model_type not in PROTO_MODELS else 'sequential',
                'val': 'sequential',
                'test': 'sequential'}
    train_kwargs = dict(data_source=args.data_source,
                          label_map=args.label_map,
                          target_col=args.target_col,
                          bag_size=args.train_bag_size,
                          shuffle=True)
    
    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                          label_map=args.label_map,
                          target_col=args.target_col,
                          bag_size=args.val_bag_size)

    all_results, all_dumps = {}, {}

    # Cross-validation
    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    model_type=args.model_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    sampler_types=sampler_types,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results, fold_dumps = train(dataset_splits, args, mode='classification')

    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    fold_dumps.update({'labels': np.array(list(args.label_map.keys()), dtype=np.object_)})
    save_pkl(dump_path, fold_dumps)

    return final_dict


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str,
                    choices=['adamW', 'sgd'], default='adamW')
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=1)

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', action='store_true',
                    default=False, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=15,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=10,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

##
# model / loss fn args ###
parser.add_argument('--model_type', type=str, choices=['H2T', 'ABMIL', 'TransMIL', 'SumMIL', 'OT', 'PANTHER', 'ProtoCount', 'DeepAttnMIL', 'ILRA'],
                    default='ABMIL',
                    help='type of model')
parser.add_argument('--emb_model_type', type=str, default='LinEmb_LR')
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_config', type=str,
                    default='ABMIL_default', help="name of model config file")

parser.add_argument('--in_dim', default=768, type=int,
                    help='dim of input features')
parser.add_argument('--in_dropout', default=0.0, type=float,
                    help='Probability of dropping out input features.')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)

parser.add_argument('--train_sampler', type=str, default='random', 
                    choices=['random', 'weighted', 'sequential'])
parser.add_argument('--n_fc_layers', type=int)
parser.add_argument('--em_iter', type=int)
parser.add_argument('--tau', type=float)
parser.add_argument('--out_type', type=str, default='param_cat')

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str, default='.')
parser.add_argument('--fix_proto', action='store_true', default=False)
parser.add_argument('--n_proto', type=int)

# experiment task / label args ###
parser.add_argument('--exp_code', type=str,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, choices=tasks)
parser.add_argument('--target_col', type=str, default='label')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    args.label_map = label_dicts[args.task]
    print('label map: ', args.label_map)
    args.n_classes = len(set(list(args.label_map.values())))
    print('task: ', args.task)
    args.split_dir = j_('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    
    split_num = args.split_dir.split('/')[2].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[2].split('_k=')[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    print(args.proto_path)
    if os.path.isfile(args.proto_path):
        args.proto_fname = '/'.join(args.proto_path.split('/')[-2:])

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ### 
        feat_name = os.path.basename(src)
        mag, patch_size = extract_patching_info(os.path.dirname(src))
        if (mag < 0 or patch_size < 0):
            raise ValueError(f"invalid patching info parsed for {src}")
        check_params_same.append([feat_name, mag, patch_size])
    
    try:
        check_params_same = pd.DataFrame(check_params_same, columns=['feats_name', 'mag', 'patch_size'])
        print(check_params_same.to_string())
        assert check_params_same.drop(['feats_name'],axis=1).drop_duplicates().shape[0] == 1
        print("All data sources have the same feature extraction parameters.")
    except:
        print("Data sources do not share the same feature extraction parameters. Exiting...")
        sys.exit()
        
    ### Updated parsed mdoel parameters in args.Namespace ###
    #### parse patching info ####
    mag, patch_size = extract_patching_info(os.path.dirname(args.data_source[0]))
    
    #### parse model name ####
    parsed = parse_model_name(feat_name) 
    parsed.update({'patch_mag': mag, 'patch_size': patch_size, 'feat_names': sorted(list(set(check_params_same['feats_name'].tolist())))})
    for key, val in parsed.items():
        setattr(args, key, val)
     
    ### setup results dir ###
    if args.exp_code is None:
        if args.model_config == 'PANTHER_default':
            exp_code = f"{args.split_name_clean}::{args.model_config}+{args.emb_model_type}::{args.loss_fn}::{feat_name}"
        else:
            exp_code = f"{args.split_name_clean}::{args.model_config}::{feat_name}"
    else:
        pass
    
    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          str(exp_code), 
                          str(exp_code)+f"::{get_current_time()}")

    os.makedirs(args.results_dir, exist_ok=True)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    print("FINISHED!\n\n\n")
