#!/bin/bash

gpuid=$1
split_dir=$2
split_names=$3
dataroots=("$@")

feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1024
mag='20x'
patch_size=256
n_sampling_patches=100000 # Number of patch features to connsider for each prototype. Total number of patch fatures = n_sampling_patches * n_proto
mode='faiss'  # 'faiss' or 'kmeans'
n_proto=16  # Number of prototypes
n_init=5  # Number of KMeans initializations to perform

# Validity check for feat paths
all_feat_dirs=""
for dataroot_path in "${dataroots[@]}"; do
  feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}_fp/${feat}/feats_pt
  if ! test -d $feat_dir
  then
    continue
  fi

  if [[ -z ${all_feat_dirs} ]]; then
    all_feat_dirs=${feat_dir}
  else
    all_feat_dirs=${all_feat_dirs},${feat_dir}
  fi
done

cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_prototype \\
--mode ${mode} \\
--data_source ${all_feat_dirs} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--in_dim ${input_dim} \\
--n_proto_patches ${n_sampling_patches} \\
--n_proto ${n_proto} \\
--n_init ${n_init} \\
--seed 1 \\
--num_workers 10 \\
"

eval "$cmd"