#  EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use
- Authors: Jan Schlüter, Gerald Gutenbrunner
- Paper: https://arxiv.org/abs/2207.05508

This is the official PyTorch implementation for our EUSIPCO 2022 paper "EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use".

## Introduction

[LEAF](https://openreview.net/forum?id=jM76BCb6F9m) is an audio frontend with Gabor filters of learnable center frequencies and bandwidths. It was proposed as an alternative to mel spectrograms, but is about 300x slower to compute. EfficientLEAF is a drop-in replacement for LEAF, only about 10x slower than mel spectrograms. We achieve this by dynamically adapting convolution filter sizes and strides, and by replacing PCEN (Per-Channel Energy Normalization) with better parallizable operations (median subtraction and temporal batch normalization). Our experiments show that EfficientLEAF is as good as LEAF, but fail to show a clear advantage over fixed mel spectrograms &ndash; hence "of questionable use". Please still feel free to try it, your mileage may vary!

This repository provides our frontend as a PyTorch module for reuse, and scripts for a complete reproduction of the experiments presented in our paper.

## Repository Organization

If you are just interested in looking at the implementation, this list may serve as a table of contents:
- [`model/__init__.py`](model/__init__.py): generic audio classifier network
- [`model/efficientleaf.py`](model/efficientleaf.py): the EfficientLEAF frontend
- [`model/leaf.py`](model/leaf.py): our implementation of LEAF
- [`model/mel.py`](model/mel.py): mel spectrogram frontend
- [`model/OG_leaf.py`](model/OG_leaf.py): a 1:1 port of LEAF from TensorFlow (see implementation notes at the end of this document)
- `datasets/`: preparation of datasets used for the experiments and respective dataloader classes
- [`engine.py`](engine.py): train and evaluation functions
- [`environment.yml`](environment.yml): used dependencies for this repository
- [`experiments.sh`](experiments.sh): bash script for the experiments performed
- [`main.py`](main.py): training and testing script for LEAF, EfficientLEAF and a fixed mel filterbank
- [`prepare_birdclef2021.sh`](prepare_birdclef2021.sh): prepares the BirdCLEF 2021 dataset
- [`utils.py`](utils.py): utility functions for moving optimizer and scheduler to a device

## Dependencies

Our model implementation requires:
* Python >= 3.8
* PyTorch >= 1.9.0
* numpy

The experiments additionally require:
* tqdm
* tensorboard
* tensorflow-datasets
* efficientnet_pytorch
* PySoundFile

[Conda](https://docs.conda.io/en/latest/) users may want to use our provided `environment.yml`. Open a terminal / Anaconda Console in the root of this repository and create the environment via:
```
conda env create -f environment.yml
```
Activate the created enviroment with:
```
activate efficientleaf
```

## Reevaluation of experiments

The repository allows to reproduce the experiments from the paper.

### Dataset preparation

Make up your mind where the datasets shall live; you will need about 182 GiB of space (76 GiB without BirdCLEF 2021). Create an empty directory and provide its location via the `--data-path=` argument. Each data loader will initially download the dataset into a subdirectory under this path, except for the BirdCLEF dataset.

For the BirdCLEF dataset, you need to register for the respective [Kaggle challenge](https://www.kaggle.com/c/birdclef-2021/), download the data and run the `prepare_birdclef2021.sh` script (pass it the directory of the downloaded dataset and the dataset directory you use as `--data-path=`).

### Running all experiments

The bash script `experiments.sh` calls `main.py` repeatedly to carry out all experiments described in the paper. You will want to pass the following arguments:
* `--data-path=` to point to an empty directory the datasets will be downloaded to
* `--device=cuda` to run on GPU
* `--cudnn-benchmark` to allow cuDNN to optimize convolutions
* `--pin-mem` to use pinned memory in data loaders
* `--num-workers=4` to use multiprocessing in data loaders

If you have multiple GPUs, you may want to specify a GPU via the `CUDA_VISIBLE_DEVICES` environment variable.

Taken together, the command line to run on the second GPU could be:
```bash
CUDA_VISIBLE_DEVICES=1 ./experiments.sh --data-path=/mnt/scratch/leaf_datasets --device=cuda  --cudnn-benchmark --pin-mem --num-workers=4
```

The script uses lockfiles to safeguard each experiment; to run multiple experiments in parallel, just call the script multiple times with different GPUs (or on different hosts sharing an NFS mount).

### Running individual experiments

For individual experiments, use the `main.py` script. It takes different arguments for general parameters (path for datasets/model/results, training/benchmarking a model, ...), the training setup (epochs, batchsize, scheduler/optimizer parameters, ...) and architecture parameters (frontend/compression type and tweaking frontend parameters).

For example, training an EfficientLeaf architecture on the SpeechCommands dataset could be run as:
```bash
python main.py --data-path "/mnt/scratch/leaf_datasets" --data-set "SPEECHCOMMANDS" --output-dir "/mnt/scratch/leaf_experiments" --device "cuda" --cudnn-benchmark --pin-mem --num-workers 4 --batch-size 256 --lr 1e-3 --scheduler --patience 10 --scheduler-factor 0.1 --min-lr 1e-5 --frontend "EfficientLeaf" --num-groups 8 --conv-win-factor 6 --stride-factor 16 --compression "TBN" --log1p-initial-a 5 --log1p-trainable --log1p-per-band --tbn-median-filter --tbn-median-filter-append --model-name "speechcommands_eleaf"
```

Other `--frontend` options are `"Leaf"` and `"Mel"`, and both can be used with a `--compression` of `"PCEN"` or `"TBN"`.

Benchmarking the frontend throughput (forward and backward pass) of any configuration can be done by appending `--frontend-benchmark` to the above command line.

Currently the parameters for the number of filters, sample rate, window length and window stride are fixed, but can be changed at the top of the `main.py` file:
```python
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0
```

## Reusing the frontend

When running a Python session from the repository root, an EfficientLEAF frontend can be initialized with:
```python
from model.efficientleaf import EfficientLeaf

frontend = EfficientLeaf(n_filters=80, min_freq=60, max_freq=7800,
                         sample_rate=16000,
                         num_groups=8, conv_win_factor=6, stride_factor=16)
```
This corresponds to the configuration "Gabor 8G-opt" from the paper. A smaller convolution window factor or a larger stride factor will be even faster, but too extreme settings will produce artifacts in the generated spectrograms.

Alternatively, the `main.py` script can be used to extract a trained or newly initialized network by appending the command `--ret-network` (with `--data-set "None"` it returns the network without dataloaders). This returns the entire network (frontend and EfficientNet backend), with access to the frontend via `network.frontend`.

## Implementation notes

We started off by porting the official [TensorFlow LEAF implementation](https://github.com/google-research/leaf-audio) to PyTorch, module by module, verifying that we get numerically identical output. This implementation is preserved in [`model/OG_leaf.py`](model/OG_leaf.py). We then modified the implementation in the following:
* The original implementation initializes the Gabor filterbank by computing a mel filterbank matrix, then [measuring the center and width](model/OG_leaf.py#L183) of each triangular filter in this (discretized) matrix. We instead [compute the center frequency and bandwidth](model/leaf.py#L12) analytically.
* The original implementation computes the real and complex responses as interleaved channels, we compute them as two blocks instead.
* We simplified the PCEN implementation to a single self-contained module.
* The original implementation learns PCEN parameters in linear space and does not guard the delta parameter to be nonnegative. We optionally learn parameters in log space (as in the original PCEN paper) instead. This was needed for training on longer inputs, as the original implementation regularly crashed with NaN in this case.

We verified that none of these changes affected classification performance after training.

## Citation

Please cite our paper if you use this repository in a publication:
```
@INPROCEEDINGS{2022eleaf,
author={Schl{\"u}ter, Jan and Gutenbrunner, Gerald},
  booktitle={Proceedings of the 30th European Signal Processing Conference (EUSIPCO)},
  title={{EfficientLEAF}: A Faster {LEarnable} Audio Frontend of Questionable Use},
  year=2022,
  month=sep}
```
