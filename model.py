import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Part of codes are borrowed from SGFormer and DeepGraphGO
# --------------------------------------------------------


class NodeUpdateLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(NodeUpdateLayer, self).__init__()

        self.ppi_linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node):
        outputs = self.ppi_linear(node.data['ppi_out'])
        outputs = self.dropout(F.relu(self.bn(outputs)))
        if 'res' in node.data:
            outputs = outputs + node.data['res']
        return {'h': outputs}

    def reset_parameters(self):
        self.ppi_linear.reset_parameters()
        self.bn.reset_parameters()


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn=2, dropout=0.5):
        super(GraphConv, self).__init__()

        self.input = nn.EmbeddingBag(in_channels, hidden_channels, mode='sum', include_last_offset=True)
        self.input_bias = nn.Parameter(torch.zeros(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        self.bn = nn.BatchNorm1d(hidden_channels)

        self.gnn_out_layers = nn.ModuleList(NodeUpdateLayer(hidden_channels, hidden_channels, dropout)
                                            for _ in range(num_gcn))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        self.bn.reset_parameters()
        for gnn_out_layer in self.gnn_out_layers:
            gnn_out_layer.reset_parameters()

    def forward(self, nf: dgl.NodeFlow, inputs):
        nf.copy_from_parent()

        outputs = self.dropout(F.relu(self.bn(self.input(*inputs) + self.input_bias)))
        nf.layers[0].data['h'] = outputs

        for i, gnn_out_layer in enumerate(self.gnn_out_layers):
            nf.block_compute(i,
                             dgl.function.u_mul_e('h', 'self', out='m_res'),
                             dgl.function.sum(msg='m_res', out='res'))
            nf.block_compute(i,
                             dgl.function.u_mul_e('h', 'ppi', out='ppi_m_out'),
                             dgl.function.sum(msg='ppi_m_out', out='ppi_out'), gnn_out_layer)
        return nf.layers[-1].data['h']


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()

    def forward(self, query_input, source_input):
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)

        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        final_output = attn_output.mean(dim=1)  # [N, D]

        return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1, dropout=0.5,
                 use_bn=True, use_residual=True, use_act=True):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.convs = nn.ModuleList()

        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(self.hidden_channels))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(self.in_channels, self.hidden_channels))

        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x


class GTPLMGO(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5,
                 gnn_num_layers=1, gnn_dropout=0.5,
                 seq_weight=1.0):

        super().__init__()

        self.trans_conv = TransConv(in_channels=in_channels, hidden_channels=hidden_channels,
                                    num_layers=trans_num_layers, num_heads=trans_num_heads, dropout=trans_dropout)

        self.graph_conv = GraphConv(in_channels=in_channels, hidden_channels=hidden_channels,
                                    num_gcn=gnn_num_layers, dropout=gnn_dropout)

        self.gnn_num_layers = gnn_num_layers

        self.seq_weight = seq_weight

        self.hidden_channels = 3 * hidden_channels

        # seqvec
        self.seq_linear = nn.Linear(1024, hidden_channels)
        self.seq_bn = nn.BatchNorm1d(hidden_channels)

        self.mlp = nn.Linear(self.hidden_channels, out_channels)

        # transformer params
        self.params1 = list(self.trans_conv.parameters())
        # gcn params
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.seq_linear.parameters()))
        self.params2.extend(list(self.seq_bn.parameters()))
        self.params2.extend(list(self.mlp.parameters()))

    def forward(self, nf: dgl.NodeFlow, batch_x, batch_seq, batch_neighbor_x):
        x_trans = self.trans_conv(batch_x)

        x_gnn = self.graph_conv(nf, batch_neighbor_x)

        x = torch.cat((x_trans, x_gnn), dim=1)

        x_seq = F.dropout(F.relu(self.seq_bn(self.seq_linear(batch_seq))), p=0.25, training=True)

        x = self.mlp(torch.cat((x, self.seq_weight * x_seq), dim=1))

        return x

    # 参数初始化
    def reset_parameters(self):
        self.graph_conv.reset_parameters()
        self.trans_conv.reset_parameters()
        self.seq_linear.reset_parameters()
        self.seq_bn.reset_parameters()
        self.mlp.reset_parameters()
