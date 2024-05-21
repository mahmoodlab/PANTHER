# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if sys.version_info < (3,):
    import string
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def pad_sequences(sequences, pre_padding=0, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0., pre_value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths) + 2*pre_padding

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        pre_pad = np.ones(sample_shape).astype(dtype) * pre_value
        pre_pad = np.repeat([pre_pad], pre_padding, axis=0)
        s = np.concatenate([pre_pad, s, pre_pad])
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def pad_profiles(sequences, pre_padding=0, maxlen=None, dtype='float32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths) + 2*pre_padding

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        pre_pad = [value]*pre_padding
        s = np.hstack([pre_pad, s, pre_pad])
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def augment(df, noise=0.1, quantity=10, max_index=4):
    def add_noise(seq):
        new_seq = seq.copy()
        mask = np.random.rand(len(seq)) < noise
        new_seq[mask] = np.random.randint(1, max_index+1, size=np.count_nonzero(mask), dtype='int32')
        return new_seq
    df_list = [df]
    for i in range(quantity - 1):
        new_df = df.copy()
        new_df['seq_index'] = df['seq_index'].apply(add_noise)
        df_list.append(new_df)
    return pd.concat(df_list)


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, noise=0.0, max_index=4):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.noise = noise
        self.max_index = max_index + 1

    def __getitem__(self, index):
        if self.noise == 0.:
            return self.data_tensor[index], self.target_tensor[index]
        data_tensor = self.data_tensor[index].clone()
        noise_mask = torch.ByteTensor([0])
        if np.random.rand() < 0.5:
            noise_mask = torch.ByteTensor([1])
            # mask = torch.Tensor(data_tensor.size(0)).uniform_() < self.noise
            mask = torch.rand_like(data_tensor, dtype=torch.float) < self.noise
            data_tensor[mask] = torch.LongTensor(
                mask.sum().item()).random_(1, self.max_index)
        return data_tensor, self.target_tensor[index], noise_mask

    def __len__(self):
        return self.data_tensor.size(0)

    def augment(self, noise=0.1, quantity=10):
        if noise <= 0.:
            return
        new_tensor = [self.data_tensor]
        for i in range(quantity - 1):
            t = self.data_tensor.clone()
            mask = torch.rand_like(t, dtype=torch.float) < noise
            t[mask] = torch.LongTensor(mask.sum().item()).random_(1, self.max_index)
            new_tensor.append(t)
        self.data_tensor = torch.cat(new_tensor)
        self.target_tensor = self.target_tensor.repeat(quantity)


def matrix_sqrt(M, normalize=True):
    if normalize:
        d = np.sqrt(np.diag(M))
        M = M / d[:, np.newaxis] / d[np.newaxis, :]
    u, v = np.linalg.eigh(M)
    return (v * np.sqrt(u)).dot(v.T)