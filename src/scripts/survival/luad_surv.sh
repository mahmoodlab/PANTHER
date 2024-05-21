#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'path/to/tcga_luad'
)


task='LUAD_survival'
target_col='dss_survival_days'
split_names='train,test'

split_dir='survival/TCGA_LUAD_overall_survival_k=0'
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"