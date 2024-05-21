# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import torch
# from Bio import SeqIO
from collections import defaultdict
from sklearn.model_selection import train_test_split

from .data_helper import pad_sequences, TensorDataset, augment


def import_DLS2FSVM(filename, delimiter='\t', delimiter2=' ',comment='>',skiprows=0, start=0, end = 0,target_col = 1, dtype=np.float32):
    # Open a file
    file = open(filename, "r")
    #print "Name of the file: ", file.name
    if skiprows !=0:
       dataset = file.read().splitlines()[skiprows:]
    if skiprows ==0 and start ==0 and end !=0:
       dataset = file.read().splitlines()[0:end]
    if skiprows ==0 and start !=0:
       dataset = file.read().splitlines()[start:]
    if skiprows ==0 and start !=0 and end !=0:
       dataset = file.read().splitlines()[start:end]
    else:
       dataset = file.read().splitlines()
    #print dataset
    newdata = []
    for i in range(0,len(dataset)):
        line = dataset[i]
        if line[0] != comment:
           temp = line.split(delimiter,target_col)
           feature = temp[target_col]
           label = temp[0]
           if label == 'N':
               label = 0
           fea = feature.split(delimiter2)
           newline = []
           newline.append(int(label))
           for j in range(0,len(fea)):
               if fea[j].find(':') >0 :
                   (num,val) = fea[j].split(':')
                   newline.append(float(val))
            
           newdata.append(newline)
    data = np.array(newdata, dtype=dtype)
    file.close()
    return data


def std(m):
    m -= m.mean(axis=1, keepdims=True)
    m /= np.linalg.norm(m, axis=1, keepdims=True).clip(1e-06)
    return m

def load_features_labels(datadir, seq_list, label_file=None):
    feature_dir = datadir + '/Feature_aa_ss_sa'
    pssm_dir = datadir + '/PSSM_Fea'
    sequencefile = datadir + '/{}.list'.format(seq_list)
    sequence_list = pd.read_csv(sequencefile, delimiter='\t', header=None)
    sequence_name = list(sequence_list[0])

    try:
        features = np.load(datadir + '/{}.features.npz'.format(seq_list))
        labels = np.load(datadir + '/{}.labels.npz'.format(seq_list))
        return sequence_name, features, labels
    except:
        pass
    if label_file is not None:
        label_index = pd.read_csv(datadir + '/' + label_file, delimiter='\t', index_col='Fold') 

    features = {}
    labels = {}
    for i in range(len(sequence_name)):
        pdb = sequence_name[i]
        print(i)
        ori_pdb = pdb
        if '.' in ori_pdb:
            pdb = pdb.replace('.', '_')
        if '_' in ori_pdb:
            pdb = pdb.replace('_', '.')

        try:
            featurefile = feature_dir + "/{}.fea_aa_ss_sa".format(pdb)
            pssmfile = pssm_dir + "/{}.pssm_fea".format(pdb)
            featuredata = import_DLS2FSVM(featurefile)
            pssmdata = import_DLS2FSVM(pssmfile)
        except:
            featurefile = feature_dir + "/{}.fea_aa_ss_sa".format(ori_pdb)
            pssmfile = pssm_dir + "/{}.pssm_fea".format(ori_pdb)
            featuredata = import_DLS2FSVM(featurefile)
            pssmdata = import_DLS2FSVM(pssmfile)

        # featuredata = import_DLS2FSVM(featurefile)
        # pssmdata = import_DLS2FSVM(pssmfile) 
        pssm_fea = pssmdata[:,1:]
        
        fea_len = (featuredata.shape[1]-1)//(20+3+2)
        # print(fea_len)
        train_labels = featuredata[:,0]
        if label_file is not None:
            fold = sequence_list[3].iloc[i]
            train_labels = label_index.loc[fold].values
        # print(train_labels)
        train_feature = featuredata[:,1:]
        train_feature_seq = train_feature.reshape(fea_len,25)
        train_feature_aa = train_feature_seq[:,0:20]
        train_feature_ss = train_feature_seq[:,20:23]
        train_feature_sa = train_feature_seq[:,23:25]
        train_feature_pssm = pssm_fea.reshape(fea_len,20)

        # print(std(train_feature_aa))
        # print(std(train_feature_pssm))
        train_feature_aa = std(train_feature_aa)
        train_feature_ss = std(train_feature_ss)
        train_feature_sa = std(train_feature_sa)
        train_feature_pssm = std(train_feature_pssm)
        ### reconstruct feature, each residue represent aa,ss,sa,pssm
        featuredata_all = np.concatenate((
            train_feature_aa, train_feature_ss, train_feature_sa, train_feature_pssm), axis=1)
        # featuredata_all = featuredata_all.reshape(1,featuredata_all.shape[0]*featuredata_all.shape[1])
        # featuredata_all_tmp = np.concatenate((train_labels.reshape((1,1)),featuredata_all), axis=1)
        # print(featuredata_all)
        # print(featuredata_all.shape)
        features[ori_pdb] = featuredata_all
        labels[ori_pdb] = int(train_labels)

    print("saving data...")
    np.savez_compressed(datadir + '/{}.features'.format(seq_list), **features)
    np.savez_compressed(datadir + '/{}.labels'.format(seq_list), **labels)
    return sequence_name, features, labels


def load_data(datadir, seq_list, pre_padding=0, maxlen=None, pre_value=0.0, label_file=None):
    sequence_name, features, labels = load_features_labels(datadir, seq_list, label_file)
    features = [features[seq] for seq in sequence_name]
    labels = np.asarray([labels[seq] for seq in sequence_name])
    features = pad_sequences(
        features, pre_padding=pre_padding, maxlen=maxlen, padding='post',
        truncating='post', dtype='float32', pre_value=pre_value)
    features = np.transpose(features, (0, 2, 1))
    print(features.shape)
    print(labels.shape)
    features, labels = torch.from_numpy(features), torch.from_numpy(labels)
    return TensorDataset(
        features, labels, noise=0.0, max_index=features.shape[-1])
