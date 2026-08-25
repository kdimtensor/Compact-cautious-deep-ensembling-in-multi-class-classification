# Compact cautious deep ensembling in multi-class classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)

We tackle visible challenges in deep ensemble learning, where deep neural networks serve as ensemble members: training and storage burdens, and robustness of cautious (set-valued) predictions targeting multiple utilities, which may involve reward-sensitivity. To mitigate the training and storage burdens, we propose to employ compact ensembles, such as Bayesian Neural Networks and Convolutional Neural Networks with the Monte-Carlo dropout prediction option, to produce probabilistic predictions. For each query instance, these probabilistic predictions are then used to define a representative distribution optimizing some statistical distance. The representative distribution is then employed to define the Bayes-optimal prediction of any utility. To address the potential unrobustness of singleton prediction making, we propose a family of set-utilities satisfying some desirable properties and whose set-valued Bayes-optimal predictions can be found efficiently. 

## Summary

The processing pipeline has three stages and each stage must be run from inside its own
directory (all paths in the code are relative to the script's directory):

```
data_preprocessing/  →  model_training/  →  optimal_prediction/
add noise               train models        make predictions
split 3-folds           save models         
```

### 0. Environment

```bash
# conda (recommended — reproduces the exact environment: Python 3.10, torch 2.8)
conda env create -f environment.yml

# or pip, into an existing Python 3.10+ environment
pip install -r requirements.txt
```

A CUDA GPU is expected (the shipped shell scripts set `CUDA_VISIBLE_DEVICES`),
but every entry point falls back to CPU when `torch.cuda.is_available()` is
`False`.

### 1. Make noisy datasets and split cross-validation folds

```bash
cd data_preprocessing

bash cifar_data_preparation.sh    # CIFAR-10   (downloads the raw data)
bash fmnist_data_preparation.sh   # Fashion-MNIST (downloads the raw data)
bash leaf_data_preparation.sh     # LEAF (raw images must already be in ../data/LEAF/)
```

Each script runs `make_<dataset>_c.py` (adds Gaussian noise, saves clean/noisy
`.npy` pairs) followed by `split_fold_<dataset>.py` (writes the 3-fold indices). See [Datasets](#datasets) for what
lands on disk.

### 2. Train and save models

```bash
cd model_training

# Bayesian neural network
bash train_bayesian_cifar.sh      # then
bash test_bayesian_cifar.sh

# Convolutional neural network with Monte-Carlo dropout
bash train_cnndropout_cifar.sh    # then
bash test_cnndropout_cifar.sh

# Bayesian neural network
bash train_bayesian_fmnist.sh      # then
bash test_bayesian_fmnist.sh

# Convolutional neural network with Monte-Carlo dropout
bash train_cnndropout_fmnist.sh    # then
bash test_cnndropout_fmnist.sh

# Bayesian neural network
bash train_bayesian_leaf.sh      # then
bash test_bayesian_leaf.sh

# Convolutional neural network with Monte-Carlo dropout
bash train_cnndropout_leaf.sh    # then
bash test_cnndropout_leaf.sh
```

Training loops over the 3 folds
and keeps, per fold, the models.

`test_*.sh` reloads a checkpoint and runs `--num_monte_carlo` (number of ensemble members, default 100) forward passes over the held-out fold then writes the ensemble outputs to disk:

| File (per fold, in `data/<DATA>-C/gaussian_<train_type>_fold/<Bayes\|Drop_out>/`) | Shape |
| --- | --- |
| `tensor_output_{c,clean}_fold_{k}.npy` | `[num_monte_carlo, N_fold, n_class]` |
| `test_labels_{c,clean}_fold_{k}.npy` | `[N_fold]` |
| `test_labels_onehot_{c,clean}_fold_{k}.npy` | `[N_fold, n_class]` |

`c` = corrupted test stream, `clean` = clean test stream; one `test` run
produces both.

Useful flags (the shell scripts are thin wrappers — edit them or call the
Python entry points directly):

| Flag | Where | Meaning |
| --- | --- | --- |
| `--mode train\|test` | all | required, train or test a model |
| `--train-type clean\|noise` | all | train on the clean or the corrupted data |
| `--k-fold` | all | number of cross validation folds (default 3, must match step 1) |
| `--arch` / `--model_name` | Bayesian / dropout | select a base learner, default `resnet20` / `resnet18` |
| `--epochs`, `--lr`, `--batch-size` (`--batch_size` for the dropout scripts) | all | optimization settings |
| `--num_mc` / `--num_monte_carlo` | all | number of ensemble members during training / at inference |
| `--p` | dropout only | dropout rate |

### 3. Precise prediction

```bash
cd optimal_prediction

python cifar_precise_prediction.py --model_type Bayes --train_type clean
python fmnist_precise_prediction.py --model_type Bayes --train_type clean
python leaf_precise_prediction.py --model_type Bayes --train_type clean
```

For each test instance, outputs of the ensemble members are aggregated into a single probability
distribution with three distances — squared Euclidean distance, L1 distance, and KL divergence. The distribution is saved as `all_p_star_{SED,L1,KLD}_{c,clean}_fold_{k}.npy` next to the tensors from
step 2.

`--model_type {Bayes,Drop_out}` and `--train_type {clean,noise}` are used to select the type of ensemble and the data it was trained with.

### 4. Set-valued (cautious) prediction

```bash
python cifar_set_valued_prediction.py --model_type Bayes --train_type clean --test_type clean
python cifar_set_valued_prediction.py --model_type Bayes --train_type clean --test_type c
python fmnist_set_valued_prediction.py --model_type Drop_out --train_type noise --test_type clean
python leaf_set_valued_prediction.py --model_type Drop_out --train_type noise --test_type c
```

Set-valued predictions can be made with or without reward sensitivity. Use --model_type to change between Bayes/Drop_out, --train_type to select models trained with clean/noise dataset, and --test_type to test the models with clean/c (clean or noisy dataset). Results
are averaged over the 3 folds and written as `mean : std` strings to:

```
data/<DATA>-C/gaussian_<train_type>_fold/<model_type>/set_output/
    <test_type>_set_value_prediction.csv   # one column per α
    <test_type>_set_value_prediction.txt   # the α values that were swept
```

## Datasets

Three image dataset are used, each in a clean and a
Gaussian-corrupted version of the same images with the same labels,
so that a model can be trained on one stream and evaluated on both.

| Dataset | Source | Classes | Input | Working root |
| --- | --- | --- | --- | --- |
| CIFAR-10 | `torchvision.datasets.CIFAR10`, downloaded automatically into `data/` | 10 | 32×32 RGB | `data/CIFAR-10-C/` |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST`, downloaded automatically into `data/` | 10 | 28×28 greyscale | `data/FMNIST-10-C/` |
| LEAF (plant-leaf disease) | **manual download**, see below | 39 | resized to 224×224 RGB (LANCZOS) | `data/LEAF-C/` |

LEAF is not downloadable from the code: place it yourself as an
`ImageFolder` tree, one sub-directory per class, before running
`leaf_data_preparation.sh` (before step 1):

```
data/LEAF/
├── <class_0>/*.jpg
├── <class_1>/*.jpg
└── ...            # 39 class directories
```

`make_leaf_c.py` exits with a message if `data/LEAF/` is missing, and writes the
class order it discovered to
`data/LEAF-C/gaussian_noise_1_6/output.txt` (Fashion-MNIST likewise gets a
`label_list.txt`) — the label indices used everywhere downstream.

**Corruption.** A single corruption type is used: additive Gaussian noise,
`clip(x/255 + N(0, σ), 0, 1) · 255`, with
`σ ∈ {0.04, 0.06, 0.08, 0.09, 0.10}` for severities 1…5.
`make_cifar_c.py` / `make_fmnist_c.py` generate and stack all five severities
into `gaussian_noise_1_6/`; `split_fold_*.py` then keeps only the last block —
the strongest noise, `σ = 0.10`, referred to as *severity 6* in the directory
names — and writes it to `gaussian_noise_6/`. `make_leaf_c.py` skips the stack
and generates that severity directly.

**Folds.** The clean and corrupted copies share one index space, so a single
`KFold(n_splits=3, shuffle=True, random_state=42)` over the pooled
train+test set gives the indices used by every model and every stage. They are
stored once, in `data/<DATA>-C/k_fold_id/`, and re-loaded by the training,
testing and evaluation scripts.

| | stacked (5 severities) | kept (severity 6) | pooled for CV | per fold (train / test) |
| --- | --- | --- | --- | --- |
| CIFAR-10 | 250,000 train + 50,000 test | 50,000 + 10,000 | 60,000 | 40,000 / 20,000 |
| Fashion-MNIST | 300,000 train + 50,000 test | 60,000 + 10,000 | 70,000 | ≈46,667 / ≈23,333 |
| LEAF | — (severity 6 only) | all images (train split only) | all images | 2/3 / 1/3 |

Resulting layout (everything under `data/` is generated and should stay out of
version control):

```
data/
├── cifar-10-batches-py/ , FashionMNIST/ , LEAF/     # raw sources
└── <CIFAR-10-C | FMNIST-10-C | LEAF-C>/
    ├── gaussian_noise_1_6/          # all severities stacked (intermediate, large)
    ├── gaussian_noise_6/            # the severity actually used
    │   ├── train/ , test/           # corrupted images  (.npy, uint8)
    │   ├── train_clean/ , test_clean/
    │   └── label/label_{train,test}.npy
    ├── k_fold_id/{train,test}_ids_fold_{0,1,2}.npy
    └── gaussian_<clean|noise>_fold/ # ← written by steps 2-4
        ├── Bayes/{model/, *.csv, tensor_output_*.npy, all_p_star_*.npy, set_output/}
        └── Drop_out/{...}
```

## Repository layout

```
main/
├── environment.yml                     # exact conda environment (Python 3.10, torch 2.8)
├── requirements.txt                    # minimal pip dependencies
│
├── data_preprocess/                    # Step 1 — corruption + cross-validation folds
│   ├── <cifar|fmnist|leaf>_data_preparation.sh   # runs the two scripts below
│   ├── make_cifar_c.py                 # Gaussian noise, severities 1-5, clean/noisy .npy pairs
│   ├── make_fmnist_c.py
│   ├── make_leaf_c.py                  # ImageFolder → 224×224, severity 6 only
│   ├── split_fold_cifar.py             # keep severity 6 + KFold(3, seed 42) → k_fold_id/
│   ├── split_fold_fmnist.py
│   ├── split_fold_leaf.py
│   ├── cifar10_c.py                    # CIFAR10C / CIFAR10CLEAN  (VisionDataset over the .npy)
│   ├── fmnist_c.py                     # FMNISTC  / FMNISTCLEAN
│   └── leaf_c.py                       # LEAFC    / LEAFCLEAN
│
├── trained_model/                      # Step 2 — training + Monte-Carlo inference
│   ├── main_bayesian_cifar.py          # variational ResNet-20 (bayesian-torch), ELBO = CE + KL/batch
│   ├── main_bayesian_fmnist.py         # variational SCNN (bayesian-torch)
│   ├── main_bayesian_leaf.py           # variational ResNet [3,3,3], 39 classes
│   ├── main_cnndropout_cifar.py        # MC-dropout ResNet-18
│   ├── main_cnndropout_fmnist.py       # MC-dropout SimpleDropout [2,2,2,2]
│   ├── main_cnndropout_leaf.py         # MC-dropout ResNet [2,2,2,2], 39 classes
│   ├── {train,test}_bayesian_{cifar,fmnist,leaf}.sh
│   ├── {train,test}_cnndropout_{cifar,fmnist,leaf}.sh
│   ├── models/
│   │   ├── resnet.py                   # deterministic ResNet
│   │   └── resnet_dropout.py           # ResNet18Dropout / SimpleDropout / ResNetDropout_leaf
│   ├── contrib/                        # vendored ADF layers + lgamma extension (unused by default)
│   ├── saved_loss.py                   # per-epoch loss/accuracy → CSV
│   └── utils.py                        # dataset mean/std, init helpers
│
├── Optimal_prediction/                 # Steps 3-4 — decision making on the credal set
│   ├── cifar_precise_prediction.py     # p* under SED / L1 / KLD  → all_p_star_*.npy
│   ├── fmnist_precise_prediction.py
│   ├── leaf_precise_prediction.py
│   ├── cifar_set_valued_prediction.py  # α-sweep, IDC set-valued rule → set_output/*.csv
│   ├── fmnist_set_valued_prediction.py
│   └── leaf_set_valued_prediction.py
│
└── data/                               # generated by the pipeline (not tracked)
```

Every stage communicates with the next one only through `.npy` files under
`data/`, so stages can be re-run independently as long as the fold indices in
`k_fold_id/` are left untouched.

## Citation

If you use this code, please cite the paper (see
[CITATION.cff](CITATION.cff)):

```bibtex
```
## License

Released under the [MIT License](LICENSE).
