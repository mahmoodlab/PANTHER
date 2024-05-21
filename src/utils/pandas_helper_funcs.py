import os
from os.path import join as j_
from os import listdir as ldir_
from os import scandir as sdir_
import shutil

from tqdm import tqdm

import numpy as np
import pandas
import pandas as pd
from pandas import Series

def series_int(s1: pandas.Series, s2: pandas.Series, dtype='O'):
    """
    Returns set intersection of two pd.Series (resets index).

    Args:
        s1 (pandas.Series): Series (of strings).
        s2 (pandas.Series): Series (of strings).

    Returns:
        (pandas.Series): set interesection of s1 and s2
    """
    return pd.Series(list(set(s1) & set(s2)), dtype=dtype)

def df_loc_col(df1: pandas.DataFrame, s1: pandas.Series, col_name: str, apply_sint=True, drop_orig_index=False):
    """
    Performs pandas.loc with column <col_name> as index of df1.

    Args:
        df1 (pandas.DataFrame): Dataframe.
        s1 (pandas.Series): Series (of strings).
        col_name (str): column to use as index for pandas.loc
        apply_sint (bool): Whether to take series intersection first.
        drop_orig_index (bool): Drops original index.

    Returns:
       (pandas.DataFrame): df1 subsetted by s1 using column <col_name>.
    """
    return df1.reset_index(drop=drop_orig_index).set_index(col_name).loc[series_int(df1[col_name], s1) if apply_sint else s1].reset_index(drop=False)

def series_ldir_int(path1, path2, exts=['.', '.'], add_ext=False):
    """
    Gets intersection of fnames (accounting for differing exts) in path1 and path2.

    Args:
        path1 (_type_): Path to directory of fnames.
        path2 (_type_): Path to directory of fnames.
        exts (list): Which exts to split for fnames in path1 and path2. Defaults to ['.', '.'].
        add_ext (bool, optional): Whether to add back in the extension. 
            If True, defaults to using extension of path1 (order matters). Defaults to False.

    Returns:
        (pd.Series): Intersection of fnames (accounting for differing exts) in path1 and path2
    """
    df1 = pd.Series(ldir_(path1)).str.rsplit(pat=exts[0], n=1, expand=True).set_axis(['fname', 'ext'], axis=1)
    df2 = pd.Series(ldir_(path2)).str.rsplit(pat=exts[1], n=1, expand=True).set_axis(['fname', 'ext'], axis=1)
    df = df_loc_col(df1, df2['fname'], col_name='fname', apply_sint=True, drop_orig_index=True)
    return df['fname']+exts[0]+df['ext'] if add_ext else df['fname']

def df_sdir(dataroot: str, cols=['fpath', 'fname', 'slide_id']):
    """
    Returns pd.DataFrame of the file paths and fnames of contents in dataroot.

    Args:
        dataroot (str): path to files.

    Returns:
        (pandas.Dataframe): pd.DataFrame of the file paths and fnames of contents in dataroot (make default cols: ['fpath', 'fname_ext', 'fname_noext']?)
    """
    return pd.DataFrame([(e.path, e.name, os.path.splitext(e.name)[0]) for e in sdir_(dataroot)], columns=cols)


# TODO: Fix doc + also make function for ldir_diff
def series_diff(s1, s2, dtype='O'):
    r"""
    Returns set difference of two pd.Series.
    """
    return pd.Series(list(set(s1).difference(set(s2))), dtype=dtype)


def transfer_dir2dir_shutil(dataroot: str, saveroot: str, subset_fnames:str=None, lim: int=None):
    r"""
    Transfer files from dir2dir
    
    Args:
    - dataroot (str): Source folder from which you want to transfer files from
    - subset_fnames (list): List of filenames
    - saveroot (str): Destination folder from which you want to transfer files to
    
    Return:
    - None
    """
    from tqdm import tqdm
    from os.path import join as j_
    if lim == None:
        pbar = tqdm(os.listdir(dataroot))
    else:
        pbar = tqdm(os.listdir(dataroot)[:lim])
    
    missing = []
    for fname in pbar:
        pbar.set_description(f'Copying: {fname}')

        src = j_(dataroot, fname)
        dst = j_(saveroot, fname)

        if not os.path.isfile(src):
            missing.append(fname)
        elif os.path.isfile(dst):
            continue
        else:
            shutil.copyfile(src=src, dst=dst)
        pass
    
    print('Num Missing:', len(missing))
    print("Missing Files:", missing)

series_intersection = series_int
series_difference = series_diff