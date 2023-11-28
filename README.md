# LazyResampling
A reference implementation of lazy resampling for pytorch, meant to accompany "Lazy Resampling: fast and information preserving preprocessing for deep learning".

This repository is a placeholder for a standalone reference implementation. The current reference implementation used in the paper is available at [https://github.com/atbenmurray/MONAI/tree/lr_development](https://github.com/atbenmurray/MONAI/tree/lr_development).

This repository additionally contains the python notebooks and instructions for replicating the results of the experiments described in the paper.

## Installation

It is recommended that you use [conda](https://docs.conda.io/projects/miniconda/en/latest/) to create a virtual environment in which to run both the jupyter notebook and the model training. The instructions will use the environment name `lazyresampling` but feel free to use an alternative name. Please follow [these instructions](https://docs.conda.io/projects/miniconda/en/latest/) for installing miniconda if you don't already have it.

### Create the conda env
Note that other python versions may be used but these instructions have been validated for python 3.10.
```
conda create --name lazyresampling python=3.10
conda activate lazyresampling
```

### Install the Lazy Resampling package
As mentioned, the reference implementation for Lazy Resampling is temporarily a branch on a fork of MONAI, but will be moved to this repository in due course. It should be installed as follows:
```
git clone git@github.com:atbenmurray/MONAI lazyresampling
cd lazyresampling
git checkout lr_development
pip install -e .
```

### Install the reference network package

The reference network is a unet implemention installed as follows:
```
git clone git@github.com:atbenmurray/relight
cd relight
pip install -e .
```

## Training the reference network

The reference network can be trained by calling the following script:
```
python train_unet.py -S <seed_number> -L
```

`-S` sets the seed number for training; `-L` specifies that the preprocessing should run in *lazy* mode, whereas omitting it runs preprocessing in *traditional* mode.

