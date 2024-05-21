"""
All the functions related to clustering and slide embedding construction
"""

import pdb
import os
from utils.file_utils import save_pkl, load_pkl
import numpy as np
import time
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cluster(data_loader, n_proto, n_iter, n_init=5, feature_dim=1024, n_proto_patches=50000, mode='kmeans', use_cuda=False):
    """
    K-Means clustering on embedding space

    For further details on FAISS,
    https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
    """
    n_patches = 0

    n_total = n_proto * n_proto_patches

    # Sample equal number of patch features from each WSI
    try:
        n_patches_per_batch = (n_total + len(data_loader) - 1) // len(data_loader)
    except:
        n_patches_per_batch = 1000

    print(f"Sampling maximum of {n_proto * n_proto_patches} patches: {n_patches_per_batch} each from {len(data_loader)}")

    patches = torch.Tensor(n_total, feature_dim)

    for batch in tqdm(data_loader):
        if n_patches >= n_total:
            continue

        data = batch['img'] # (n_batch, n_instances, instance_dim)

        with torch.no_grad():
            out = data.reshape(-1, data.shape[-1])[:n_patches_per_batch]  # Remove batch dim

        size = out.size(0)
        if n_patches + size > n_total:
            size = n_total - n_patches
            out = out[:size]
        patches[n_patches: n_patches + size] = out
        n_patches += size

    print(f"\nTotal of {n_patches} patches aggregated")

    s = time.time()
    if mode == 'kmeans':
        print("\nUsing Kmeans for clustering...")
        print(f"\n\tNum of clusters {n_proto}, num of iter {n_iter}")
        kmeans = KMeans(n_clusters=n_proto, max_iter=n_iter)
        kmeans.fit(patches[:n_patches].cpu())
        weight = kmeans.cluster_centers_[np.newaxis, ...]

    elif mode == 'faiss':
        assert use_cuda, f"FAISS requires access to GPU. Please enable use_cuda"
        try:
            import faiss
        except ImportError:
            print("FAISS not installed. Please use KMeans option!")
            raise
        
        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
        print(f"\tNum of clusters {n_proto}, num of iter {n_iter}")

        kmeans = faiss.Kmeans(patches.shape[1], 
                              n_proto, 
                              niter=n_iter, 
                              nredo=n_init,
                              verbose=True, 
                              max_points_per_centroid=n_proto_patches,
                              gpu=numOfGPUs)
        kmeans.train(patches)
        weight = kmeans.centroids[np.newaxis, ...]

    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")

    e = time.time()
    print(f"\nClustering took {e-s} seconds!")

    return n_patches, weight

def check_prototypes(n_proto, embed_dim, load_proto, proto_path):
    """
    Check validity of the prototypes
    """
    if load_proto:
        assert os.path.exists(proto_path), "{} does not exist!".format(proto_path)
        if proto_path.endswith('pkl'):
            prototypes = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            prototypes = np.load(proto_path)


        assert (n_proto == prototypes.shape[0]) and (embed_dim == prototypes.shape[1]),\
            "Prototype dimensions do not match! Params: ({}, {}) Suplied: ({}, {})".format(n_proto,
                                                                                           embed_dim,
                                                                                           prototypes.shape[0],
                                                                                           prototypes.shape[1])

