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

class WSISurvivalDataset(Dataset):
    """WSI Survival Dataset."""

    def __init__(self,
                 df,
                 data_source,
                 target_transform=None,
                 sample_col='case_id',
                 slide_col='slide_id',
                 survival_time_col='os_survival_days',
                 censorship_col='os_censorship',
                 n_label_bins=4,
                 label_bins=None,
                 bag_size=0,
                 include_surv_t0=True,
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
        self.target_col = survival_time_col
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.include_surv_t0 = include_surv_t0

        is_nan_censorship = self.data_df[self.censorship_col].isna()
        if sum(is_nan_censorship) > 0:
            print('# of NaNs in Censorship col, dropping:', sum(is_nan_censorship))
            self.data_df = self.data_df[~is_nan_censorship]

        is_nan_survival = self.data_df[self.survival_time_col].isna()
        if sum(is_nan_survival) > 0:
            print('# of NaNs in Survival time col, dropping:', sum(is_nan_survival))
            self.data_df = self.data_df[~is_nan_survival]

        if (self.data_df[self.survival_time_col] < 0).sum() > 0 and (not self.include_surv_t0):
            self.data_df = self.data_df[self.data_df[self.survival_time_col] > 0]

        censorship_vals = self.data_df[self.censorship_col].value_counts().index
        if set(censorship_vals) != set([0,1]):
            print('Censorship values must be binary integers, found:', censorship_vals)
            sys.exit()

        self.target_transform = target_transform
        self.n_label_bins = n_label_bins
        self.label_bins = None
        self.bag_size = bag_size

        self.validate_survival_dataset()
        self.idx2sample_df = pd.DataFrame({'sample_id': self.data_df[sample_col].astype(str).unique()})
        self.set_feat_paths_in_df()
        self.data_df.index = self.data_df[sample_col].astype(str)
        self.data_df.index.name = 'sample_id'
        self.X = None
        self.y = None
        
        if 'disc_label' in self.data_df.columns:
            self.data_df = self.data_df.drop('disc_label', axis=1)
        
        if self.n_label_bins > 0:
            disc_labels, label_bins = compute_discretization(df=self.data_df,
                                                             survival_time_col=self.survival_time_col,
                                                             censorship_col=self.censorship_col,
                                                             n_label_bins=self.n_label_bins,
                                                             label_bins=label_bins)
            self.data_df = self.data_df.join(disc_labels)
            self.label_bins = label_bins
            self.target_col = disc_labels.name
            assert self.data_df.index.nunique() == self.idx2sample_df.index.nunique()

        self.survival_time_labels = []
        self.censorship_labels = []
        self.disc_labels = []
        for idx in self.idx2sample_df.index:
            survival_time, censorship, disc_label = self.get_labels(idx)
            self.survival_time_labels.append(survival_time)
            self.censorship_labels.append(censorship)
            self.disc_labels.append(disc_label)

        self.survival_time_labels = torch.tensor(self.survival_time_labels)
        self.censorship_labels = torch.tensor(self.censorship_labels)
        self.disc_labels = torch.tensor(self.disc_labels)

    def __len__(self):
        return len(self.idx2sample_df)

    def set_feat_paths_in_df(self):
        """
        Sets the feature path (for each slide id) in self.data_df. At the same time, checks that all slides 
        specified in the split (or slides for the cases specified in the split) exist within data source.
        """
        self.feats_df = pd.concat([df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col]) for feats_dir in self.data_source]).drop(['fname'], axis=1).reset_index(drop=True)
        missing_feats_in_split = series_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])

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

    def validate_survival_dataset(self):
        """Validate that the survival dataset is valid."""
        # check that each case_id has only one survival value
        num_unique_surv_times = self.data_df.groupby(self.sample_col)[self.survival_time_col].unique().apply(len)
        try:
            assert (num_unique_surv_times == 1).all()
        except AssertionError:
            print('Each case_id must have only one unique survival value.')
            raise

        # check that all survival values are numeric
        try:
            assert not pd.to_numeric(self.data_df[self.survival_time_col], errors='coerce').isna().any()
        except AssertionError:
            print('Survival values must be numeric.')
            raise

        # check that all survival values are positive
        try:
            assert (self.data_df[self.survival_time_col] >= 0).all()
            if not self.include_surv_t0:
                assert (self.data_df[self.survival_time_col] > 0).all()
        except AssertionError:
            print('Survival values must be positive.')
            raise

        # check that all censorship values are binary integers
        try:
            assert self.data_df[self.censorship_col].isin([0, 1]).all()
        except AssertionError:
            print('Censorship values must be binary integers.')
            raise

    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        feat_paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        if isinstance(feat_paths, str):
            feat_paths = [feat_paths]
        return feat_paths

    def get_labels(self, idx):
        labels = self.data_df.loc[self.get_sample_id(idx), [self.survival_time_col, self.censorship_col, self.target_col]]
        if isinstance(labels, pd.Series):
            labels = list(labels)
        elif isinstance(labels, pd.DataFrame):
            labels = list(labels.iloc[0])
        return labels

    def __getitem__from_emb__(self, idx):
        out = {'img': self.X[idx],
            'coords': [],
            'survival_time': torch.tensor([self.survival_time_labels[idx]]),
            'censorship': torch.tensor([self.censorship_labels[idx]]),
            'label': torch.tensor([self.disc_labels[idx]])}
        return out

    def __getitem__(self, idx):
        if self.X is not None:
            return self.__getitem__from_emb__(idx)
        
        survival_time, censorship, label = self.get_labels(idx)
        # Read features (and coordinates, Optional) from pt/h5 file
        all_features = []
        all_coords = []

        feat_paths = self.get_feat_paths(idx)
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
            'survival_time': torch.Tensor([survival_time]),
            'censorship': torch.Tensor([censorship]),
            'label': torch.Tensor([label])}

        if attn_mask is not None:
            out['attn_mask'] = attn_mask

        return out
    
    def get_label_bins(self):
        return self.label_bins


def compute_discretization(df, survival_time_col='os_survival_days', censorship_col='os_censorship', n_label_bins=4, label_bins=None):
    df = df[~df['case_id'].duplicated()] # make sure that we compute discretization on unique cases

    if label_bins is not None:
        assert len(label_bins) == n_label_bins + 1
        q_bins = label_bins
    else:
        uncensored_df = df[df[censorship_col] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6  # set rightmost edge to be infinite
        q_bins[0] = -1e-6  # set leftmost edge to be 0

    disc_labels, q_bins = pd.cut(df[survival_time_col], bins=q_bins,
                                retbins=True, labels=False,
                                include_lowest=True)
    assert isinstance(disc_labels, pd.Series) and (disc_labels.index.name == df.index.name)
    disc_labels.name = 'disc_label'
    return disc_labels, q_bins
