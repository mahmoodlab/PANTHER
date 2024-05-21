"""
This will construct unsupervised slide embedding

Good reference for clustering
https://github.com/facebookresearch/faiss/wiki/FAQ#questions-about-training
"""

from __future__ import print_function

import argparse
from torch.utils.data import DataLoader
from wsi_datasets import WSIProtoDataset
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from mil_models import prepare_emb
from mil_models import PrototypeTokenizer

import numpy as np

import pdb
import os
from os.path import join as j_

def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}):
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy()
        dataset = WSIProtoDataset(df, **dataset_kwargs)

        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')

    return dataset_splits


def main(args):
    
    train_kwargs = dict(data_source=args.data_source,
                        use_h5=args.use_h5)
       
    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('\nsuccessfully read splits for: ', list(csv_splits.keys()))

    dataset_splits = build_datasets(csv_splits,
                                    batch_size=1,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs)

    print('\nInit Datasets...', end=' ')
    os.makedirs(j_(args.split_dir, 'embeddings'), exist_ok=True)
    
    # Construct unsupervised slide-level embedding
    datasets, fpath = prepare_emb(dataset_splits, args, mode='emb')

    # Construct tokenized slide-level embedding
    if args.out_type == 'allcat':
        print("Generting Tokenized slide embeddings..")
        tokenizer = PrototypeTokenizer(args.model_type, args.out_type, args.n_proto)
        embeddings = {}
        for k, loader in datasets.items():
            prob, mean, cov = tokenizer(loader.dataset.X)
            embeddings[k] = {'prob': prob, 'mean': mean, 'cov': cov}
        fpath_new = fpath.rsplit('.', 1)[0] + '_tokenized.pkl'
        save_pkl(fpath_new, embeddings)


    print("\nSlide embedding construction finished!")



# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
# model / loss fn args ###
parser.add_argument('--n_proto', type=int, help='Number of prototypes')
parser.add_argument('--in_dim', type=int)
parser.add_argument('--model_type', type=str, choices=['H2T', 'OT', 'PANTHER', 'ProtoCount'],
                    help='type of embedding model')
parser.add_argument('--em_iter', type=int)
parser.add_argument('--tau', type=float)
parser.add_argument('--out_type', type=str)
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_config', type=str,
                    default='ABMIL_default', help="name of model config file")

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--use_h5', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=8)

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str)
parser.add_argument('--fix_proto', action='store_true', default=False)

args = parser.parse_args()

if __name__ == "__main__":
    args.split_dir = j_('splits', args.split_dir)
    args.split_name = os.path.basename(args.split_dir)
    print('split_dir: ', args.split_dir)

    args.data_source = [src for src in args.data_source.split(',')]

    if args.load_proto:
        assert os.path.exists(args.proto_path), f"The proto path {args.proto_path} doesn't exist!"

    results = main(args)