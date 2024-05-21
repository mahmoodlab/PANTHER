#!/bin/bash

gpuid=$1
check=$2
config=$3
wandb_project=$4

### Dataset Information
declare -a dataroots=(
	'path/to/panda'
)

task='panda'
target_col='isup_grade'
split_dir="classification/panda_wholesight"
split_names='train,val,test_K,test_R'

bash "./scripts/classification/${config}.sh" $gpuid $check $wandb_project $task $target_col $split_dir $split_names "${dataroots[@]}"
