from __future__ import print_function, division
import os
from os.path import join as j_
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import sys

from torch.utils.data import Dataset
import h5py
from .dataset_utils import apply_sampling
sys.path.append('../')
from utils.pandas_helper_funcs import df_sdir, series_diff

class WSIClassificationDataset(Dataset):
    """WSI Classification Dataset."""

    def __init__(self,
                 df,
                 data_source,
                 target_transform=None,
                 sample_col='slide_id',
                 slide_col='slide_id',
                 target_col='label',
                 label_map=None,
                 bag_size=0,
                 **kwargs):
        """
        Args:
        """
        self.data_source = []
        for src in data_source:
            assert os.path.basename(src) in ['feats_h5', 'feats_pt']
            self.use_h5 = True if os.path.basename(src) == 'feats_h5' else False
            self.data_source.append(src)

        self.data_df = df
        assert 'Unnamed: 0' not in self.data_df.columns
        self.sample_col = sample_col
        self.slide_col = slide_col
        self.target_col = target_col
        self.target_transform = target_transform
        self.label_map = label_map
        self.bag_size = bag_size
        self.data_df[sample_col] = self.data_df[sample_col].astype(str)
        self.data_df[slide_col] = self.data_df[slide_col].astype(str)
        self.X = None
        self.y = None

        self.validate_classification_dataset()
        self.idx2sample_df = pd.DataFrame({'sample_id': self.data_df[sample_col].astype(str).unique()})
        self.set_feat_paths_in_df()
        self.data_df.index = self.data_df[sample_col].astype(str)
        self.data_df.index.name = 'sample_id'
        print(self.data_df.groupby([target_col])[sample_col].count().to_string())

        self.labels = []
        for idx in self.idx2sample_df.index:
            self.labels.append(self.get_labels(idx, apply_transform=True))

        self.labels = torch.Tensor(self.labels).type(torch.long)

    def __len__(self):
        return len(self.idx2sample_df)

    def set_feat_paths_in_df(self):
        """
        Sets the feature path (for each slide id) in self.data_df. At the same time, checks that all slides 
        specified in the split (or slides for the cases specified in the split) exist within data source.
        """
        self.feats_df = pd.concat([df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col]) for feats_dir in self.data_source]).drop(['fname'], axis=1).reset_index(drop=True)
        missing_feats_in_split = series_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])

        ### Assertion to make sure there  are no unexpected labels in split
        try:
            self.data_df[self.target_col].map(self.label_map)
        except:
            print(f"Unexpected labels in split:\n{self.data_df[self.target_col].unique()}")
            sys.exit()

        ### Assertion to make sure that there are not any missing slides that were specified in your split csv file
        try:
            assert len(missing_feats_in_split) == 0
        except:
            print(f"Missing Features in Split:\n{missing_feats_in_split}")
            sys.exit()

        ### Assertion to make sure that all slide ids to feature paths have a one-to-one mapping (no duplicated features).
        try:
            self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.slide_col, validate='1:1')
            assert self.feats_df[self.slide_col].duplicated().sum() == 0
        except:
            print("Features duplicated in data source(s). List of duplicated features (and their paths):")
            print(self.feats_df[self.feats_df[self.slide_col].duplicated()].to_string())
            sys.exit()

        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]

    def validate_classification_dataset(self):
        """
        - Why is this needed? For ebrains, slides for a single case have different disease diagnoses (often the case for temporal data, or patients who undergo multiple resections).
        """
        num_unique_target_labels = self.data_df.groupby(self.sample_col)[self.target_col].unique().apply(len)
        try:
            assert (num_unique_target_labels == 1).all()
        except AssertionError:
            print('Each case_id must have only one unique survival value.')
            raise

    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        feat_paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        if isinstance(feat_paths, str):
            feat_paths = [feat_paths]
        return feat_paths

    def get_labels(self, idx, apply_transform=False):
        if isinstance(idx, int):
            idx = [idx]
        labels = self.data_df.loc[self.get_sample_id(idx), self.target_col]
        if isinstance(labels, pd.Series):
            labels = labels.values.tolist()
        if apply_transform:
            if self.label_map is not None:
                labels = [self.label_map[label] for label in labels]
            if self.target_transform is not None:
                labels = [self.target_transform(label) for label in labels]
        
        if len(idx) == 1:
            labels = labels[0]
        return labels

    def __getitem__from_emb__(self, idx):
        out = {'img': self.X[idx],
               'coords': [],
               'label': torch.Tensor([self.labels[idx]])}

        return out

    def __getitem__(self, idx):
        if self.X is not None:
            return self.__getitem__from_emb__(idx)
        
        feat_paths = self.get_feat_paths(idx)
        label = self.get_labels(idx, apply_transform=True)

        # Read features (and coordinates, Optional) from pt/h5 file
        all_features = []
        all_coords = []
        for feat_path in feat_paths:
            if self.use_h5:
                with h5py.File(feat_path, 'r') as f:
                    features = f['features'][:]
                    coords = f['coords'][:]
                all_coords.append(coords)
            else:
                features = torch.load(feat_path)

            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)
            all_features.append(features)
            
        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))
        if len(all_coords) > 0:
            all_coords = np.concatenate(all_coords, axis=0)

        # apply sampling if needed, return attention mask if sampling is applied else None
        all_features, all_coords, attn_mask = apply_sampling(self.bag_size, all_features, all_coords)

        out = {'img': all_features,
               'coords': all_coords,
               'label': label}
        if attn_mask is not None:
            out['attn_mask'] = attn_mask

        return out
