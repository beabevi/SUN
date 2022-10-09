import argparse
import multiprocessing as mp
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
from ogb.graphproppred import Evaluator
from pathlib import Path

# noinspection PyUnresolvedReferences
from data import SubgraphData
from utils import get_data, get_model, SimpleEvaluator, NonBinaryEvaluator, Evaluator
from main import reset_wandb_env, train, eval, str2bool


def run(args, device, fold_idx, sweep_run_name, sweep_id, results_queue, num_train_data):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    reset_wandb_env()
    run_name = "{}-{}".format(sweep_run_name, fold_idx)
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=args,
    )

    train_loader, train_loader_eval, valid_loader, test_loader, attributes = get_data(args, fold_idx, num_train_data)
    in_dim, out_dim, task_type, eval_metric = attributes

    if 'ogb' in args.dataset:
        evaluator = Evaluator(args.dataset)
    else:
        evaluator = SimpleEvaluator(eval_metric) if args.dataset != "IMDB-MULTI" \
                                                  and args.dataset != "CSL" else NonBinaryEvaluator(out_dim)

    model = get_model(args, in_dim, out_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if 'ZINC' in args.dataset:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    elif 'ogb' in args.dataset:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    if "classification" in task_type:
        criterion = torch.nn.BCEWithLogitsLoss() if args.dataset != "IMDB-MULTI" \
                                                    and args.dataset != "CSL" else torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.L1Loss()

    # If sampling, perform majority voting on the outputs of 5 independent samples
    voting_times = 5 if args.fraction != 1. else 1

    train_curve = []
    valid_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):

        train(model, device, train_loader, optimizer, criterion, epoch=epoch, fold_idx=fold_idx)

        train_perf = eval(model, device, train_loader_eval, evaluator, voting_times)
        valid_perf = eval(model, device, valid_loader, evaluator, voting_times)
        test_perf = eval(model, device, test_loader, evaluator, voting_times)

        if scheduler is not None:
            if 'ZINC' in args.dataset:
                scheduler.step(valid_perf[eval_metric])
                if optimizer.param_groups[0]['lr'] < 0.00001:
                    break
            else:
                scheduler.step()

        train_curve.append(train_perf[eval_metric])
        valid_curve.append(valid_perf[eval_metric])
        test_curve.append(test_perf[eval_metric])

        run.log(
            {
                f'Metric/train': train_perf[eval_metric],
                f'Metric/valid': valid_perf[eval_metric],
                f'Metric/test': test_perf[eval_metric]
            }
        )

    wandb.join()

    results_queue.put((train_curve, valid_curve, test_curve))
    return

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str,
                        help='Type of convolution {gin, originalgin, zincgin, graphconv}')
    parser.add_argument('--random_ratio', type=float, default=0.,
                        help='Number of random features, > 0 only for RNI')
    parser.add_argument('--model', type=str,
                        help='Type of model {deepsets, dss, gnn}')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--channels', type=str,
                        help='String with dimension of each DS layer, separated by "-"'
                             '(considered only if args.model is deepsets)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--jk', type=str, default="last",
                        help='JK strategy, either last or concat (default: last)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training (default: 0.01)')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate for training (default: 0.5)')
    parser.add_argument('--decay_step', type=int, default=50,
                        help='decay step for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--data_dir', type=str, default="dataset/",
                        help='directory where to store the data (default: dataset/)')
    parser.add_argument('--policy', type=str, default="edge_deleted",
                        help='Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}'
                             ' (default: edge_deleted)')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='Depth of the ego net if policy is ego_nets (default: 2)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)')
    parser.add_argument('--patience', type=int, default=20,
                        help='patience (default: 20)')
    parser.add_argument('--task_idx', type=int, default=-1,
                        help='Task idx for Counting substracture task')
    parser.add_argument('--use_transpose', type=str2bool, default=False,
                        help='Whether to use transpose in SUN')
    parser.add_argument('--use_residual', type=str2bool, default=False,
                        help='Whether to use residual in SUN')
    parser.add_argument('--test', action='store_true',
                        help='quick test')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--add_bn', type=str2bool, default=True,
                        help='Whether to use batchnorm in SUN')
    parser.add_argument('--use_readout', type=str2bool, default=True,
                        help='Whether to use subgraph readout in SUN')
    parser.add_argument('--use_mlp', type=str2bool, default=True,
                        help='Whether to use mlps (instead of linears) in SUN')
    parser.add_argument('--subgraph_readout', type=str, default='sum',
                        help='Subgraph readout, default sum')

    args = parser.parse_args()

    args.channels = list(map(int, args.channels.split("-")))
    if args.channels[0] == 0:
        # Used to get NestedGNN from DS
        args.channels = []
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mp.set_start_method('spawn')

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    if 'ogb' in args.dataset or 'ZINC' in args.dataset or 'subgraphcount' in args.dataset:
        n_folds = 1
    elif 'CSL' in args.dataset:
        n_folds = 5
    else:
        n_folds = 10

    # number of processes to run in parallel
    # TODO: make it dynamic
    if n_folds > 1 and 'REDDIT' not in args.dataset:
        if args.dataset == 'PROTEINS':
            num_proc = 2
        else:
            num_proc = 3 if args.batch_size == 128 and args.dataset != 'MUTAG' and args.dataset != 'PTC' else 5
    else:
        num_proc = 1

    if args.dataset in ['CEXP', 'EXP']:
        num_proc = 2
    if 'IMDB' in args.dataset and args.policy == 'edge_deleted':
        num_proc = 1

    all_res = pd.DataFrame(columns = [f'Metric/train_mean', f'Metric/val_mean', f'Metric/test_mean',
                                'num_train_data' ]
    )
    data_range = range(50, 1600, 100) if 'subgraphcount' in args.dataset else range(1_000, 11_000, 1_000)
    for num_train_data in data_range:
        num_free = num_proc
        results_queue = mp.Queue()

        curve_folds = []
        fold_idx = 0

        run(args, device, fold_idx, sweep_run_name, sweep_id, results_queue, num_train_data)
        curve_folds.append(results_queue.get())

        train_curve_folds = np.array([l[0] for l in curve_folds])
        valid_curve_folds = np.array([l[1] for l in curve_folds])
        test_curve_folds = np.array([l[2] for l in curve_folds])

        # compute aggregated curves across folds
        train_curve = np.mean(train_curve_folds, 0)
        train_accs_std = np.std(train_curve_folds, 0)

        valid_curve = np.mean(valid_curve_folds, 0)
        valid_accs_std = np.std(valid_curve_folds, 0)

        test_curve = np.mean(test_curve_folds, 0)
        test_accs_std = np.std(test_curve_folds, 0)

        task_type = 'classification' if args.dataset != 'ZINC' and args.dataset != 'subgraphcount' else 'regression'
        if 'classification' in task_type:
            best_val_epoch = np.argmax(valid_curve)
            best_train = max(train_curve)
        else:
            if args.dataset == 'subgraphcount':
                best_val_epoch = np.argmin(valid_curve)
            else:
                best_val_epoch = len(valid_curve) - 1
            best_train = min(train_curve)

        results = [
            train_curve[best_val_epoch],
            valid_curve[best_val_epoch],
            test_curve[best_val_epoch],
            num_train_data
        ]
        all_res.loc[len(all_res.index)] = results

    outdir = Path(f"out/{args.dataset}/{args.policy}")
    outdir.mkdir(parents=True, exist_ok=True)
    all_res.to_csv(outdir / f"{args.model}-{args.seed}.csv", index=False)

    wandb.join()

if __name__ == "__main__":
    main()