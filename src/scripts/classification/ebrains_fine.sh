#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	"path/to/ebrains"
)

task='ebrains_subtyping_fine'
target_col='diagnosis'
split_dir='classification/ebrains'
split_names='train,val,test'

bash "./scripts/classification/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
