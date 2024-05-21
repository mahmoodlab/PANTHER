#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'path/to/tcga_brca'
)

split_names='train,test'

split_dir='survival/TCGA_BRCA_overall_survival_k=0'
bash "./scripts/embedding/${config}.sh" $gpuid $split_dir $split_names "${dataroots[@]}"