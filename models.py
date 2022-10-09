import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

from conv import GNN_node


def get_subgraph_idx(batched_data):
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    graph_offset = tmp[batched_data.batch]

    subgraph_idx = graph_offset + batched_data.subgraph_batch
    return subgraph_idx


def get_root_idx(batched_data):
    num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
    # repeat for each subgraph in the graph
    num_nodes_per_subgraph = num_nodes_per_subgraph[batched_data.subgraph_idx_batch]

    subgraph_offset = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                     torch.cumsum(num_nodes_per_subgraph, dim=0)])[:-1]

    root_idx = subgraph_offset + batched_data.subgraph_idx
    return root_idx


def get_node_idx(batched_data):
    num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
    tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                        torch.cumsum(num_nodes_per_subgraph, dim=0)])
    graph_offset = tmp[batched_data.batch]
    # Same idx for a node appearing in different subgraphs of the same graph
    node_idx = graph_offset + batched_data.subgraph_node_idx
    return node_idx


def get_transpose_idx(batched_data):
    # batched_data.num_nodes_per_subgraph = torch.tensor([3, 2, 1, 2])
    # batched_data.batch = torch.tensor([
    #     0,0,0,
    #     0,0,0,
    #     0,0,0,

    #     1,1,
    #     1,1,

    #     2,

    #     3,3,
    #     3,3])
    # batched_data.subgraph_idx_batch = torch.tensor([
    #     0,
    #     0,
    #     0,

    #     1,
    #     1,

    #     2,

    #     3,
    #     3])

    # batched_data.subgraph_node_idx = torch.tensor([
    #     0,1,2,
    #     0,1,2,
    #     0,1,2,

    #     0,1,
    #     0,1,

    #     0,

    #     0,1,
    #     0,1,
    # ])

    # batched_data.subgraph_batch = torch.tensor([
    #     0,0,0,
    #     1,1,1,
    #     2,2,2,

    #     0,0,
    #     1,1,

    #     0,

    #     0,0,
    #     1,1,
    # ])

    num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
    # repeat for each node in each subgraph of the graph
    num_nodes_nod = num_nodes_per_subgraph[batched_data.batch]
    # repeat for each subgraph in the graph
    num_nodes_sub = num_nodes_per_subgraph[batched_data.subgraph_idx_batch]

    subgraph_node_idx = batched_data.subgraph_node_idx
    subgraph_batch = batched_data.subgraph_batch

    index = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                    torch.cumsum(num_nodes_per_subgraph, dim=0)])[:-1]
    subgraph_offset = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                    torch.cumsum(num_nodes_sub, dim=0)])[:-1]

    graph_offset = subgraph_offset[index][batched_data.batch]

    result = subgraph_node_idx * num_nodes_nod + subgraph_batch + graph_offset

    # assert (result == torch.tensor([ 0,  3,  6,  1,  4,  7,  2,  5,  8,  9, 11, 10, 12, 13, 14, 16, 15, 17])).all()
    # import pdb; pdb.set_trace()

    return result


def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations

    subgraph_idx = get_subgraph_idx(batched_data)

    return pool(h_node, subgraph_idx)


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 feature_encoder=lambda x: x):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = self.emb_dim if self.JK == 'last' else self.emb_dim * self.num_layer + in_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                 gnn_type=gnn_type, num_random_features=num_random_features,
                                 feature_encoder=feature_encoder)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        return subgraph_pool(h_node, batched_data, self.pool)


class GNNComplete(GNN):
    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 feature_encoder=lambda x: x):

        super(GNNComplete, self).__init__(num_tasks, num_layer, in_dim, emb_dim, gnn_type, num_random_features,
                                          residual, drop_ratio, JK, graph_pooling, feature_encoder)

        if gnn_type == 'graphconv':
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim // 2),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim // 2, out_features=num_tasks)
            )
        else:
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=num_tasks),
            )

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.final_layers(h_graph)


class DSnetwork(torch.nn.Module):
    def __init__(self, subgraph_gnn, channels, num_tasks, invariant):
        super(DSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant

        fc_list = []
        fc_sum_list = []
        for i in range(len(channels)):
            fc_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                           out_features=channels[i]))
            if self.invariant:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i],
                                                   out_features=channels[i]))
            else:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                                   out_features=channels[i]))

        self.fc_list = torch.nn.ModuleList(fc_list)
        self.fc_sum_list = torch.nn.ModuleList(fc_sum_list)

        dim = channels[-1] if len(channels) > 0 else subgraph_gnn.out_dim
        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=dim, out_features=2 * dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * dim, out_features=num_tasks)
        )

    def forward(self, batched_data):
        h_subgraph = self.subgraph_gnn(batched_data)

        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
            for layer_idx, fc_sum in enumerate(self.fc_sum_list):
                h_graph = F.elu(fc_sum(h_graph))
        else:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)
                x2 = fc_sum(
                    torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
                )

                h_subgraph = F.elu(x1 + x2[batched_data.subgraph_idx_batch])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)


class DSSnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv):
        super(DSSnetwork, self).__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = feature_encoder

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.feature_encoder(x)
        for i in range(len(self.gnn_list)):
            gnn, bn, gnn_sum, bn_sum = self.gnn_list[i], self.bn_list[i], self.gnn_sum_list[i], self.bn_sum_list[i]

            h1 = bn(gnn(x, edge_index, edge_attr))

            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                             torch.cumsum(num_nodes_per_subgraph, dim=0)])
            graph_offset = tmp[batch]

            # Same idx for a node appearing in different subgraphs of the same graph
            node_idx = graph_offset + batched_data.subgraph_node_idx
            x_sum = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")

            h2 = bn_sum(gnn_sum(x_sum, batched_data.original_edge_index,
                                batched_data.original_edge_attr if edge_attr is not None else edge_attr))

            x = F.relu(h1 + h2[node_idx])

        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)


class SUNnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv,
                    use_transpose=False, drop_ratio=0., res=False, add_bn=True,
                    use_readout=True, use_mlp=True, subgraph_readout="sum"):
        super(SUNnetwork, self).__init__()

        self.emb_dim = emb_dim
        self.use_transpose = use_transpose
        self.drop_ratio = drop_ratio
        self.res = res
        self.use_readout = use_readout
        if self.res:
            self.lin1 = torch.nn.Linear(in_dim, emb_dim)
        if subgraph_readout == "sum":
            self.subgraph_readout = global_add_pool
        elif subgraph_readout == "mean":
            self.subgraph_readout = global_mean_pool
        else:
            raise ValueError("Subgraph readout must be sum or mean.")

        self.feature_encoder = feature_encoder

        gnn_list = []
        gnn_root_list = []
        gnn_sum_list = []
        gnn_root_sum_list = []
        bn_list = []
        bn_root_list = []
        bn_sum_list = []
        bn_root_sum_list = []
        u_readout_list = []
        u_readout_root_list = []
        u_vv_list = []
        u_kk_list = []
        u_kv_list = []
        u_vk_list = []
        u_vv_root_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            gnn_root_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim) if add_bn else torch.nn.Identity())
            bn_root_list.append(torch.nn.BatchNorm1d(emb_dim) if add_bn else torch.nn.Identity())

            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            gnn_root_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim) if add_bn else torch.nn.Identity())
            bn_root_sum_list.append(torch.nn.BatchNorm1d(emb_dim) if add_bn else torch.nn.Identity())

            for l in [u_readout_list, u_readout_root_list, u_vv_list, u_kk_list, u_kv_list, u_vv_root_list]:
                if use_mlp:
                    net = torch.nn.Sequential(torch.nn.Linear(emb_dim if i != 0 else in_dim, emb_dim),
                                                torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                                torch.nn.Linear(emb_dim, emb_dim)
                    )
                else:
                    net = torch.nn.Sequential(torch.nn.Linear(emb_dim if i != 0 else in_dim, emb_dim))
                l.append(net)

            if self.use_transpose:
                u_vk_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim if i != 0 else in_dim, emb_dim),
                                                        torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                                        torch.nn.Linear(emb_dim, emb_dim)
                ))
        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_root_list = torch.nn.ModuleList(gnn_root_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)
        self.gnn_root_sum_list = torch.nn.ModuleList(gnn_root_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_root_list = torch.nn.ModuleList(bn_root_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)
        self.bn_root_sum_list = torch.nn.ModuleList(bn_root_sum_list)

        self.u_readout_list = torch.nn.ModuleList(u_readout_list)
        self.u_readout_root_list = torch.nn.ModuleList(u_readout_root_list)

        self.u_vv_list = torch.nn.ModuleList(u_vv_list)
        self.u_kk_list = torch.nn.ModuleList(u_kk_list)
        self.u_kv_list = torch.nn.ModuleList(u_kv_list)
        self.u_vk_list = torch.nn.ModuleList(u_vk_list)
        self.u_vv_root_list = torch.nn.ModuleList(u_vv_root_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_ratio),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        root_idx = get_root_idx(batched_data) # ids of root nodes
        subgraph_idx = get_subgraph_idx(batched_data) # for each node which subgraph it belogs to
        node_idx = get_node_idx(batched_data) # same idx for a node in different subgraphs of the same graph

        if self.use_transpose:
            transpose_idx = get_transpose_idx(batched_data)

        is_root = torch.zeros_like(node_idx).bool()
        is_root[root_idx] = 1

        x = self.feature_encoder(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        previous_x = x
        if self.res:
            previous_x = self.lin1(previous_x)
        for i in range(len(self.gnn_list)):
            gnn, gnn_root, gnn_sum, gnn_root_sum = self.gnn_list[i], self.gnn_root_list[i], self.gnn_sum_list[i], self.gnn_root_sum_list[i]
            u_readout, u_readout_root = self.u_readout_list[i], self.u_readout_root_list[i]
            u_vv = self.u_vv_list[i]
            u_kk = self.u_kk_list[i]
            u_kv = self.u_kv_list[i]
            u_vv_root = self.u_vv_root_list[i]

            bn, bn_root, bn_sum, bn_root_sum = self.bn_list[i], self.bn_root_list[i], self.bn_sum_list[i], self.bn_root_sum_list[i]

            # aggregate to obtain a representation of each subgraph
            h_subgraph = subgraph_pool(x, batched_data, self.subgraph_readout)

            # obtain needle
            x_sum = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")

            ####
            #  Update non-root nodes x^k_v
            ####
            root_repr = x[root_idx] # representations of root nodes

            x_vv = u_vv(root_repr)[node_idx] # get x^v_v
            x_kk = u_kk(root_repr)[subgraph_idx] # get x^k_k
            x_kv = u_kv(x) # get x^k_v

            if self.use_transpose:
                u_vk = self.u_vk_list[i]
                x_vk = u_vk(x[transpose_idx]) # get x^v_k

            readout = u_readout(h_subgraph)

            h1 = gnn(x, edge_index, edge_attr)
            h1[~is_root] = bn(h1[~is_root])

            h2 = bn_sum(gnn_sum(x_sum, batched_data.original_edge_index,
                                batched_data.original_edge_attr if edge_attr is not None else edge_attr
                                ))

            out = x_vv + x_kk + x_kv + h1 + h2[node_idx]

            if self.use_readout:
                out = out + readout[subgraph_idx]

            if self.use_transpose:
                out = out + x_vk

            ####
            #  Update root nodes x^v_v
            ####
            x_vv = u_vv_root(root_repr) # get x^v_v same as x^k_k same as x^k_v (same as x^v_k)

            readout_root = u_readout_root(h_subgraph)

            h1_root = bn_root(gnn_root(x, edge_index, edge_attr)[root_idx])

            h2_root = bn_root_sum(gnn_root_sum(x_sum, batched_data.original_edge_index,
                                        batched_data.original_edge_attr if edge_attr is not None else edge_attr
                                ))

            out[root_idx] = x_vv + h1_root + h2_root

            if self.use_readout:
                out[root_idx] = out[root_idx] + readout_root

            x = F.relu(out)
            x = F.dropout(x, self.drop_ratio, training=self.training)

            if self.res:
                x = x + previous_x
                previous_x = x

        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)


class EgoEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus' or self.policy == 'node_marked':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())
