#!/bin/bash

quants=( q4_0 q4_1 q5_0 q5_1 q8_0 q8_1 q2k q3k q4k q5k q6k q8k f16 )

# Get model name
read -p "Enter model name: " model_name

# Collect dirs
read -p "Enter path to directory with .safetensors: " safetensor_dir
read -p "Enter output directory (default ${safetensor_dir}): " output_dir
output_dir=${output_dir:=${safetensor_dir}}

# Report
echo "safetensor_dir: $safetensor_dir"
echo "output_dir: $output_dir"

# Concat all safetensors into string for final command
safetensors=`ls ${safetensor_dir} | grep .safetensors | grep -v .json | xargs`
safetensor_string=""
for s in ${safetensors[@]}; do
    safetensor_string+="${safetensor_dir}/${s} "
done

echo ${safetensor_string}

# Create All
for q in ${quants[@]}; do
    ./tensor-tools quantize --quantization ${q} \
        ${safetensor_string} \
        --out-file ${output_dir}/${model_name}_${q}.gguf
done