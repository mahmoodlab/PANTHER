#!/bin/bash

gpuid=$1
split_dir=$2
split_names=$3
dataroots=("$@")

model_tuple='OT,default'
feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1024
mag='20x'
patch_size=256

bag_size='-1'
out_size=16
em_step=1
eps=1
out_type='allcat'
load_proto=1
tau=1.0
proto_num_samples='1.0e+05'

save_dir_root=results
IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')

# Identify feature paths
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

cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_embedding \\
--data_source ${all_feat_dirs} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--model_type ${model} \\
--model_config ${model}_default \\
--in_dim ${input_dim} \\
--seed 1 \\
--num_workers 8 \\
--em_iter ${em_step} \\
--tau ${tau} \\
--n_proto ${out_size} \\
--out_type ${out_type} \\
--ot_eps ${eps} \\
--fix_proto \\
"

if [[ $load_proto -eq 1 ]]; then
  cmd="$cmd --load_proto \\
  --proto_path "splits/${split_dir}/prototypes/prototypes_c${out_size}_extracted-${feat_name}_faiss_num_${proto_num_samples}.pkl" \\
  "
fi


eval "$cmd"