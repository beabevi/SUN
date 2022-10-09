import numpy as np
from graph_property import GraphPropertyDataset
from ogb.graphproppred import Evaluator as Evaluator_
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.nn import GraphConv
from torch_geometric.transforms import OneHotDegree
from pathlib import Path

from conv import GINConv, GraphConvE, OriginalGINConv, GCNConv, ZINCGINConv, ZINCGraphConvE
from counting_data import GraphCountDataset
from csl_data import MyGNNBenchmarkDataset
# noinspection PyUnresolvedReferences
from data import policy2transform, preprocess, SubgraphData, TUDataset, PTCDataset, Sampler
from gnn_rni_data import PlanarSATPairsDataset
from models import GNN, GNNComplete, DSnetwork, DSSnetwork, EgoEncoder, SUNnetwork, ZincAtomEncoder


def get_data(args, fold_idx, num_train_data=None):
    if args.model == 'gnn': assert args.policy == 'original'

    root = Path(args.data_dir)
    if args.policy in ["ego_nets", "ego_nets_plus"]:
        root /= str(args.num_hops)

    transform = Sampler(args.fraction)

    # automatic dataloading and splitting
    if 'ogb' in args.dataset:
        dataset = PygGraphPropPredDataset(root=root / args.policy,
                                          name=args.dataset,
                                          pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops),
                                          )
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        split_idx = dataset.get_idx_split()

    elif args.dataset == 'PTC':
        dataset = PTCDataset(root=root / args.policy,
                             name=args.dataset,
                             pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops),
                             )
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        split_idx = dataset.separate_data(args.seed, fold_idx=fold_idx)

    elif args.dataset == 'CSL':
        dataset = MyGNNBenchmarkDataset(root=root / args.policy,
                                        name=args.dataset,
                                        pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops,
                                                                       process_subgraphs=OneHotDegree(5)
                                                                       ))
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        split_idx = dataset.separate_data(args.seed, fold_idx=fold_idx)

    elif args.dataset == 'ZINC':
        dataset = ZINC(root=root / args.policy, subset=True, split="train")
        val_dataset = ZINC(root=root / args.policy, subset=True, split="val")
        test_dataset = ZINC(root=root / args.policy, subset=True, split="test")

    elif args.dataset in ['CEXP', 'EXP']:
        dataset = PlanarSATPairsDataset(root=root / args.policy,
                                        name=args.dataset,
                                        pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops))
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        split_idx = dataset.separate_data(args.seed, fold_idx=fold_idx)

    elif args.dataset == 'subgraphcount':
        dataset = GraphCountDataset(root=root / args.policy,
                            name=args.dataset,
                            pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops),
                            )
        assert args.task_idx != -1
        dataset.data.y = dataset.data.y[:, args.task_idx: args.task_idx + 1]
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        # ensure edge_attr is not considered
        dataset.data.edge_attr = None
        split_idx = dataset.separate_data(args.seed, fold_idx=fold_idx)


    elif args.dataset == 'graphproperty':
        dataset = GraphPropertyDataset(root=root / args.policy / args.dataset, split="train")
        val_dataset = GraphPropertyDataset(root=root / args.policy / args.dataset, split="val")
        test_dataset = GraphPropertyDataset(root=root / args.policy / args.dataset, split="test")
        assert args.task_idx != -1

        dataset.data.y = dataset.data.y[:, args.task_idx: args.task_idx + 1]
        val_dataset.data.y = val_dataset.data.y[:, args.task_idx: args.task_idx + 1]
        test_dataset.data.y = test_dataset.data.y[:, args.task_idx: args.task_idx + 1]
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        # ensure edge_attr is not considered
        dataset.data.edge_attr = None

    else:
        dataset = TUDataset(root=root / args.policy,
                            name=args.dataset,
                            pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops),
                            )
        if args.fraction != 1.:
            dataset = preprocess(dataset, transform)
        # ensure edge_attr is not considered
        dataset.data.edge_attr = None
        split_idx = dataset.separate_data(args.seed, fold_idx=fold_idx)

    train_ds = dataset[split_idx["train"]] if args.dataset != 'ZINC' and args.dataset != "graphproperty" else dataset
    train_ds = train_ds if num_train_data is None else train_ds[:num_train_data]
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, follow_batch=['subgraph_idx'])
    train_loader_eval = DataLoader(train_ds,
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, follow_batch=['subgraph_idx'])
    valid_loader = DataLoader(dataset[split_idx["valid"]] if args.dataset != 'ZINC' and args.dataset != "graphproperty" else val_dataset,
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, follow_batch=['subgraph_idx'])
    test_loader = DataLoader(dataset[split_idx["test"]] if args.dataset != 'ZINC' and args.dataset != "graphproperty" else test_dataset,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, follow_batch=['subgraph_idx'])

    if 'ogb' in args.dataset or 'ZINC' in args.dataset:
        in_dim = args.emb_dim if args.policy != "ego_nets_plus" and args.policy != "node_marked" else args.emb_dim + 2
    elif args.dataset == 'CSL':
        in_dim = 6 if args.policy != "ego_nets_plus" and args.policy != "node_marked" else 6 + 2  # used deg as node feature
    else:
        in_dim = dataset.num_features
    out_dim = dataset.num_tasks if args.dataset != 'ZINC' else 1

    task_type = 'regression' if args.dataset == 'ZINC' else dataset.task_type
    eval_metric = 'mae' if args.dataset == 'ZINC' else dataset.eval_metric
    return train_loader, train_loader_eval, valid_loader, test_loader, (in_dim, out_dim, task_type, eval_metric)


def get_model(args, in_dim, out_dim, device):
    encoder = lambda x: x
    if 'ogb' in args.dataset:
        encoder = AtomEncoder(args.emb_dim) if args.policy != "ego_nets_plus" and args.policy != "node_marked" else EgoEncoder(AtomEncoder(args.emb_dim))
    elif 'ZINC' in args.dataset:
        encoder = ZincAtomEncoder(policy=args.policy, emb_dim=args.emb_dim)

    if args.model == 'deepsets':

        subgraph_gnn = GNN(gnn_type=args.gnn_type, num_tasks=out_dim, num_layer=args.num_layer, in_dim=in_dim,
                           emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, JK=args.jk,
                           graph_pooling='sum' if args.gnn_type != 'gin' and len(args.channels) > 0 else 'mean', feature_encoder=encoder
                           ).to(device)
        model = DSnetwork(subgraph_gnn=subgraph_gnn, channels=args.channels, num_tasks=out_dim,
                          invariant=args.dataset == 'ZINC').to(device)

    elif args.model == 'dss':

        if args.gnn_type == 'gin':
            GNNConv = GINConv
        elif args.gnn_type == 'originalgin':
            GNNConv = OriginalGINConv
        elif args.gnn_type == 'graphconv':
            GNNConv = GraphConv
        elif args.gnn_type == 'gcn':
            GNNConv = GCNConv
        elif args.gnn_type == 'zincgin':
            GNNConv = ZINCGINConv
        else:
            raise ValueError('Undefined GNN type called {}'.format(args.gnn_type))

        model = DSSnetwork(num_layers=args.num_layer, in_dim=in_dim, emb_dim=args.emb_dim, num_tasks=out_dim,
                           feature_encoder=encoder, GNNConv=GNNConv).to(device)

    elif args.model == 'sun':

        if args.gnn_type == 'gin':
            GNNConv = GINConv
        elif args.gnn_type == 'originalgin':
            GNNConv = OriginalGINConv
        elif args.gnn_type == 'graphconv':
            GNNConv = GraphConv
        elif args.gnn_type == 'graphconvedge':
            GNNConv = GraphConvE
        elif args.gnn_type == 'zincgraphconvedge':
            GNNConv = ZINCGraphConvE
        elif args.gnn_type == 'gcn':
            GNNConv = GCNConv
        elif args.gnn_type == 'zincgin':
            GNNConv = ZINCGINConv
        else:
            raise ValueError('Undefined GNN type called {}'.format(args.gnn_type))

        model = SUNnetwork(num_layers=args.num_layer, in_dim=in_dim, emb_dim=args.emb_dim, num_tasks=out_dim,
                           feature_encoder=encoder, GNNConv=GNNConv, use_transpose=args.use_transpose,
                           drop_ratio=args.drop_ratio, res=args.use_residual, add_bn=args.add_bn,
                           use_readout=args.use_readout, use_mlp=args.use_mlp, subgraph_readout=args.subgraph_readout).to(device)

    elif args.model == 'gnn':
        num_random_features = int(args.random_ratio * args.emb_dim)
        model = GNNComplete(gnn_type=args.gnn_type, num_tasks=out_dim, num_layer=args.num_layer, in_dim=in_dim,
                            emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, JK=args.jk,
                            graph_pooling='sum' if args.gnn_type != 'gin' else 'mean',
                            feature_encoder=encoder, num_random_features=num_random_features,
                            ).to(device)

    else:
        raise ValueError('Undefined model type called {}'.format(args.model))

    return model


class SimpleEvaluator():
    def __init__(self, eval_metric):
        self.eval_metric = eval_metric

    def acc(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = (np.concatenate(y_pred, axis=-1) > 0.).astype(int)
        y_pred = (np.mean(y_pred, axis=-1) > 0.5).astype(int)

        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return {'acc': sum(acc_list) / len(acc_list)}

    def mae(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = np.concatenate(y_pred, axis=-1)
        y_pred = np.mean(y_pred, axis=-1)

        return {'mae': np.average(np.abs(y_true - y_pred))}

    def log_mse(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = np.concatenate(y_pred, axis=-1)
        y_pred = np.mean(y_pred, axis=-1)

        return {'log_mse': np.log10(np.average(np.square(y_true - y_pred)))}

    def eval(self, input_dict):
        if self.eval_metric == 'acc': return self.acc(input_dict)
        if self.eval_metric == 'mae': return self.mae(input_dict)
        return self.log_mse(input_dict)


class NonBinaryEvaluator():
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks

    def eval(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = np.concatenate(y_pred, axis=-1)
        y_pred = y_pred.argmax(1)
        y_pred = np.eye(self.num_tasks)[y_pred]
        y_pred = y_pred.sum(1).argmax(1)

        is_labeled = y_true == y_true
        correct = y_true[is_labeled] == y_pred[is_labeled]

        return {'acc': float(np.sum(correct)) / len(correct)}


class Evaluator(Evaluator_):
    def eval(self, input_dict):
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        y_pred = np.concatenate(y_pred, axis=-1)
        y_pred = np.mean(y_pred, axis=-1)

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return super().eval(input_dict)
