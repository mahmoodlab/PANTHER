#!/bin/bash

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")

feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1024
mag='20x'
patch_size=256

model_tuple='ABMIL,default'
bag_size=-1
batch_size=1
lr_scheduler='cosine'
opt='adamW'
max_epoch=20
lr=0.0001
wd=0.00001
save_dir_root='results'
							
IFS=',' read -r model config_suffix <<< "${model_tuple}"

model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')
save_dir=${save_dir_root}/${task}/${task}::${model_config}

th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
	warmup=0
	curr_lr_scheduler='constant'
else
	curr_lr_scheduler=$lr_scheduler
	warmup=1
fi

if [[ $bag_size == "-1" ]]; then
	curr_bag_size=$bag_size
	curr_batch_size=1
	grad_accum=32
else
	if [[ $patch_size == 512 ]]; then
		curr_bag_size='1024'
	else
		curr_bag_size=$bag_size
	fi
	curr_batch_size=$batch_size
	grad_accum=0
fi

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

cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_classification \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--accum_steps ${grad_accum} \\
--task ${task} \\
--target_col ${target_col} \\
--model_type ${model} \\
--model_config ${model_config} \\
--in_dim ${input_dim} \\
--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--warmup_epochs ${warmup} \\
--max_epochs ${max_epoch} \\
--train_bag_size ${curr_bag_size} \\
--batch_size ${curr_batch_size} \\
--early_stopping \\
--seed 1 \\
--in_dropout 0 \\
--num_workers 8 \\
"

eval "$cmd"