#!/bin/bash

gpuid=$1

declare -a dataroots=(
	'path/to/tcga_brca'
)

# Loop through different folds
for k in 0; do
	split_dir="survival/TCGA_BRCA_overall_survival_k=${k}"
	split_names="train"
	bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"
done