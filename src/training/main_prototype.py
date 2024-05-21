"""
This will perform K-means clustering on the training data

Good reference for clustering
https://github.com/facebookresearch/faiss/wiki/FAQ#questions-about-training
"""

from __future__ import print_function

import argparse
import torch
from torch.utils.data import DataLoader
from wsi_datasets import WSIProtoDataset
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from utils.proto_utils import cluster

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
    os.makedirs(j_(args.split_dir, 'prototypes'), exist_ok=True)

    loader_train = dataset_splits['train']
    
    _, weights = cluster(loader_train,
                            n_proto=args.n_proto,
                            n_iter=args.n_iter,
                            n_init=args.n_init,
                            feature_dim=args.in_dim,
                            mode=args.mode,
                            n_proto_patches=args.n_proto_patches,
                            use_cuda=True if torch.cuda.is_available() else False)
    

    save_fpath = j_(args.split_dir,
                    'prototypes',
                    f"prototypes_c{args.n_proto}_{args.data_source[0].split('/')[-2]}_{args.mode}_num_{args.n_proto_patches:.1e}.pkl")

    save_pkl(save_fpath, {'prototypes': weights})


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
# model / loss fn args ###
parser.add_argument('--n_proto', type=int, help='Number of prototypes')
parser.add_argument('--n_proto_patches', type=int, default=10000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5,
                    help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50,
                    help='Number of iterations for Kmeans clustering')
parser.add_argument('--in_dim', type=int)
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--use_h5', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

if __name__ == "__main__":
    args.split_dir = j_('splits', args.split_dir)
    args.split_name = os.path.basename(args.split_dir)
    print('split_dir: ', args.split_dir)

    args.data_source = [src for src in args.data_source.split(',')]
    results = main(args)