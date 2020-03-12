# Profiling a PyTorch Language Model with pyprof

Simple reference for profiling language model in PyTorch.

## Usage

1. Create a python virtual environment. e.g.: 
```
python3 -m venv josiah
source josiah/bin/activate
```
2. Install CUDA 10.1 ([link](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal))

(getting errors with CUDA 10.2).

3. Install apex ([link](https://github.com/NVIDIA/apex))

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
4. Install requirements

`pip install -r requirements.txt`

5. Profile using pyprof ([link](https://github.com/NVIDIA/apex/tree/master/apex/pyprof)). e.g.,

```
nvprof -f -o net.sql --profile-from-start off -- python language-modeling.py
python -m apex.pyprof.parse net.sql > net.dict
python -m apex.pyprof.prof -w 100 -c kernel,op,sil,tc,flops,bytes,device,stream,block,grid net.dict
python -m apex.pyprof.prof --csv -c kernel,mod,op,dir,sil,tc,flops,bytes,device,stream,block,grid net.dict > lm.csv
```

## References

This repo contains code which has been sourced from two examples:
- Language Model on the official [PyTorch website](https://github.com/pytorch/examples/tree/master/word_language_model).
- Lenet example on [pyprof github](https://github.com/NVIDIA/apex/blob/master/apex/pyprof/examples/lenet.py)
