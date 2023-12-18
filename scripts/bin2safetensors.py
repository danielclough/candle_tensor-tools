import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os

# Collect dirs
bin_dir = input("Enter path to directory with .bin: ")
output_dir = input(f"Enter output directory (default {bin_dir}): ") or bin_dir

# Report
print(f"bin_dir: {bin_dir}")
print(f"output_dir: {output_dir}")

# Concat all bins into string for final command
bins = [s for s in os.listdir(bin_dir) if s.endswith(".bin")]

# Convert .bin to .safetensors
for bin_file in bins:
    path=os.path.join(bin_dir, bin_file)
    print("\nloading ", path)
    model = torch.load(path)
    new_file = bin_file.replace(".bin", ".safetensors").replace("pytorch_", "")
    print(f"loaded {bin_file}\n\tlength: {len(model)}\n")
    save_file(model, f"{os.path.join(output_dir, new_file)}")
