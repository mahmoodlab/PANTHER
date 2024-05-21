import pickle
import h5py
from os.path import join as j_
import numpy as np
import pandas as pd
import pdb

def save_pkl(filename, save_object):
	with open(filename,'wb') as f:
	    pickle.dump(save_object, f)

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            if data_type == np.object_: data_type = h5py.string_dtype(encoding='utf-8')
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]

            try:
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            except:
                 print(f"Error encoding {key} of dtype {data_type} into hdf5")
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path