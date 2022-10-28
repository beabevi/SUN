# Understanding and Extending Subgraph GNNs by Rethinking Their Symmetries

This repository contains the official code of the paper
**[Understanding and Extending Subgraph GNNs by Rethinking Their Symmetries](https://arxiv.org/abs/2206.11140) (NeurIPS 2022 Oral)**.

The code builds on top of the [ESAN framework](https://github.com/beabevi/ESAN).

<p align="center">
<img src=./SUN.png>
</p>

## Install

First create a conda environment
```
conda env create -f environment.yml
```
and activate it
```
conda activate subgraph
```
Then, [set-up wandb](https://docs.wandb.ai/quickstart#1.-set-up-wandb).

## Reproduce ogbg-molhiv and ZINC results

We provide the hyperparameter configurations to obtain the reported results on ogbg-molhiv (Table 2) and ZINC (Table 1).

Prepare the data
```bash
python data.py --dataset ZINC --policies ego_nets ego_nets_plus
python data.py --dataset ogbg-molhiv --policies ego_nets_plus
```

Obtain a sweep id `<sweep-id>` by running
```bash
wandb sweep configs/deterministic/<config-name>
````
where `configs/deterministic/<config-name>` is one between `configs/deterministic/SUN-ogbg-molhiv.yaml` and `configs/deterministic/SUN-ZINC.yaml`.

Run the 10 seeds with
```bash
wandb agent <sweep-id>
```
and compute mean and std of `Metric/test_mean` over the runs in the sweep to obtain SUN results in Tables 1, 2.

## Reproduce other results

First, prepare the data. Run
```bash
python data.py --dataset $DATASET --policies $POLICY
```
where `$DATASET` is one of the following:
* TUDatasets (MUTAG, PTC, PROTEINS, NCI1, NCI109, IMDB-BINARY, IMDB-MULTI) - Table 4
* graphproperty - Table 5
* subgraphcount (aka counting substructures) - Table 1

and `$POLICY` is one of the following:
* ego_nets
* ego_nets_plus
* node_marked
* null

To perform hyperparameter tuning, make use of `wandb`:

1. In `configs/deterministic` folder, choose the `yaml` file corresponding to the dataset of interest, say `<config-name>`.
    This file contains the hyperparameters grid.

2. Run
    ```bash
    wandb sweep configs/deterministic/<config-name>
    ````
    to obtain a sweep id `<sweep-id>`

3. Run the hyperparameter tuning with
    ```bash
    wandb agent <sweep-id>
    ```
    You can run the above command multiple times on each machine you would like to contribute to the grid-search

4. Open your project in your wandb account on the browser to see the results:
    * For the TUDatasets refer to `Metric/valid_mean` and `Metric/valid_std` to obtain the results.

    * For graphproperty and subgraphcount,
    compute mean and std of `Metric/train_mean`, `Metric/valid_mean`, `Metric/test_mean` by grouping over all hyperparameters and averaging over the different seeds.
    Then, take the results corresponding to the configuration obtaining the best validation metric.


Note that in `configs/deterministic/SUN-subgraphcount.yaml`,
key `task_idx` indicates the target, that is, 0, 1, 2, 3 indicates respectively Triangle, Tailed Tri., Star and 4-Cycle.
Similarly in `configs/deterministic/SUN-graphproperty.yaml`, key `task_idx` 0, 1, 2 indicates respectively IsConnected, Diameter, Radius.

## Get generalisation curves

Values for GIN and GNN-AK models are obtained with the [GNN-AK](https://github.com/LingxiaoShawn/GNNAsKernel) code; DSS-GNN, DS-GNN and NGNN values can be obtained by running the code in this repo with the appropriate model.

We report results for these methods in the `out/` folder.

SUN curves can be obtained as detailed below.

### 4-Cycles (EGO) (Figure 4a)

Prepare the data
```bash
python data.py --dataset subgraphcount --policies ego_nets
```
Run
```bash
for i in {1..10}; do python plot.py --batch_size=128 --channels=96 --dataset=subgraphcount --drop_ratio=0 --emb_dim=110 --epochs=250 --gnn_type=originalgin --jk=concat --learning_rate=0.001 --model=sun --num_layer=5 --policy=ego_nets --task_idx=3 --seed="$i"; done
```

Then, plot the curve in `ego_nets-plot.pdf` by running
```bash
python make_plot.py --policy ego_nets
```

### 4-Cycles (EGO+) (Figure 4b)

Prepare the data
```bash
python data.py --dataset subgraphcount --policies ego_nets_plus
```
Run
```bash
for i in {1..10}; do python plot.py --batch_size=128 --channels=96 --dataset=subgraphcount --drop_ratio=0 --emb_dim=96 --epochs=250 --gnn_type=originalgin --jk=concat --learning_rate=0.001 --model=sun --num_layer=6 --policy=ego_nets_plus --task_idx=3 --seed="$i"; done
```

Then, plot the curve in `ego_nets_plus-plot.pdf` by running
```bash
python make_plot.py --policy ego_nets_plus
```

### ZINC (Figure 4c)

Prepare the data
```bash
python data.py --dataset ZINC --policies ego_nets
```

Run
```bash
for i in {1..10}; do python plot.py --batch_size=128 --channels=96 --dataset=ZINC --drop_ratio=0 --emb_dim=64 --epochs=400 --gnn_type=zincgin --learning_rate=0.001 --model=sun --num_layer=6 --patience=40 --policy=ego_nets --num_hops=3 --seed="$i"; done
```

Then, plot the curve in `ego_nets-ZINC-plot.pdf` by running
```bash
python make_plot_zinc.py
```


## Credits

For attribution in academic contexts, please cite

```
@inproceedings{frasca2022understanding,
title={Understanding and Extending Subgraph GNNs by Rethinking Their Symmetries},
author={Frasca, Fabrizio and Bevilacqua, Beatrice and Bronstein, Michael M and Maron, Haggai},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
}
```
