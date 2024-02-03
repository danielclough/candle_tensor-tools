# Tensor Tools

> `tensor-tools` pre-compiled for Ubuntu (x86) so you don't have to compile the rust in order to use the scripts.
> 
> The code comes from the HuggingFace/Candle-Core [tensor-tools example](https://github.com/huggingface/candle/blob/main/candle-core/examples/tensor-tools.rs).

## Usage

### Init HF Repo

> Script requires new repo and ssh setup (see below).

The model name in this script will become your output directory for following scripts.

```sh
bash scripts/init-repo.sh
```

#### Create Repo (on HuggingFace)

https://hf.co/new

#### SSH Key Setup

```sh
# On your system
ssh-keygen -t ed25519 -C "comment here"
```

Add SSH public key to [user settings](https://huggingface.co/settings/keys)


### bin2safetensors

If you have `.bin` (pickle) files first you need to convert them to safetensors in order to quantize.

This requires python:
```sh
# Dependencies
sudo apt install python3 python3-pip python-is-python3
pip install torch safetensors numpy
# Run script
python scripts/bin2safetensors.py
```

### Quantization

If you have `.safetensors` files.

To make one of each quantization type use:
```sh
bash scripts/make-all.sh
```

To make individual quantization types follow this format:
```sh
./tensor-tools quantize --quantization <quant_type> \
  <list of .safetensors files> \
  --out-file ./Candle_model_<quant_type>.gguf
```

## Links

[Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

[Candle Types](https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/k_quants.rs)
