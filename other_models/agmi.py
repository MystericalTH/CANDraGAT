import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    global_max_pool as gmp,
    GatedGraphConv
)

import torch.nn as nn
import torch

class NativeAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(NativeAttention, self).__init__()
        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        res_i = torch.cat([h_i, h_j], dim=1)
        res_j = self.net_j(h_j)
        return torch.nn.Softmax(dim=1)(self.net_i(res_i)) * res_j

class AttnPropagation(torch.nn.Module):
    def __init__(self, gate_layer, feat_size=19, gather_width=64):
        super(AttnPropagation, self).__init__()

        self.gather_width = gather_width
        self.feat_size = feat_size

        self.gate = gate_layer

        net_i = nn.Sequential(
            nn.Linear(self.feat_size * 2, self.feat_size),
            nn.Softsign(),
            nn.Linear(self.feat_size, self.gather_width),
            nn.Softsign(),
        )
        net_j = nn.Sequential(
            nn.Linear(self.feat_size, self.gather_width), nn.Softsign()
        )
        self.attention = NativeAttention(net_i, net_j)

    def forward(self, data, index, is_eval=False):
        # propagtion
        h_0 = data
        h_1 = self.gate(h_0, index, is_eval=is_eval)
        h_1 = self.attention(h_1, h_0)

        return h_1
    
    def init_weights(self):
        self.gate.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
    
    def update_buffer(self, edge_attr, edge_index):
        self.gate.update_buffer(edge_attr, edge_index)
        

class AGMIDRPer(torch.nn.Module):
    def __init__(self,
                 in_channel,
                 drug_encoder,
                 genes_encoder,
                 neck,
                 head,
                 attn_head=None,
                 gather_width=32):
        super().__init__()

        # body
        self.drug_encoder = DrugGATEncoder()
        self.genes_encoder = AttnPropagation(feat_size=in_channel, gather_width=gather_width, gate_layer=build_component(genes_encoder))
        self.head = build_component(head)
        self.neck = build_component(neck)
        if attn_head is not None:
            self.attn_head = build_component(attn_head)
        else:
            self.attn_head = None
        

    def forward(self, data):
        """Forward function.

        Args:
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
            :param test_mode:
            :param gt:
            :param data:
        """
        x_cell = data.x_cell
        _, channels = x_cell.shape
        x_d, x_d_edge_index, x_d_batch = \
            data.x, data.edge_index, data.batch
        batch_size = x_d_batch.max().item() + 1
        # print(batch_size)
        
        x_cell_embed = self.genes_encoder(
            x_cell, None
        )
        x_cell_embed = x_cell_embed.view(batch_size, 18498, -1)
        x_cell_embed = torch.transpose(x_cell_embed, 1, 2)
        x_cell_embed = self.neck(x_cell_embed)
        # if self.attn_head is not None:
        #     x_cell_embed, A = self.attn_head(x_cell_embed)
        #     A = torch.mean(A, dim=0)
        # else:
        # A = None
        drug_embed = self.drug_encoder(x_d, x_d_edge_index, x_d_batch)
        output = self.head(x_cell_embed, drug_embed)
        return output



    def init_weights(self):
        self.drug_encoder.init_weights()
        self.genes_encoder.init_weights()
        self.neck.init_weights()
        self.head.init_weights()
        if self.attn_head is not None:
            self.attn_head.init_weights()

class DrugGATEncoder(torch.nn.Module):
    def __init__(self, 
                    num_features_xd=78, 
                    heads=10, 
                    output_dim=128, 
                    gat_dropout=0.2):
        super().__init__()

        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=heads, dropout=gat_dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=gat_dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        return x
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

class MultiEdgeGatedGraphConv(GatedGraphConv):
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                bias: bool = True, num_edges: int = None, include_edges=None,
                **kwargs):
        super().__init__(out_channels=out_channels, num_layers=num_layers, bias=bias,
                                                    aggr=aggr, **kwargs)

        self.register_buffer('cell_edge_attr', None)
        self.register_buffer('cell_edge_attr_test', None)
        self.register_buffer('cell_edge_index', None)
        self.register_buffer('cell_edge_index_test', None)

        self.multi_edges_rnn = torch.nn.GRUCell(out_channels * num_edges, out_channels, bias=bias)
        
        if include_edges is None:
            include_edges = ['ppi', 'gsea', 'pcc']
        self.include_edges = include_edges
        self.allow_edges = ['ppi', 'gsea', 'pcc']
        
    def update_buffer(self, edge_attr, edge_index):
        print('update buffer')
        assert edge_attr.shape[-1] == edge_index.shape[-1], 'edge_attribution and edge_index are not compatible'
        device = next(self.parameters()).device
        self.cell_edge_attr = edge_attr.to(device)
        self.cell_edge_index = edge_index.to(device)
        
    
    def init_weights(self):
        self.multi_edges_rnn.reset_parameters()
    
    

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, is_eval=False) -> Tensor:

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                            'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        weights = self.cell_edge_attr
        edges = self.cell_edge_index

        for i in range(self.num_layers):
            x_cur = torch.matmul(x, self.weight[i])
            if self.multi_edges_rnn is not None:
                m = self.propagate(edges, x=x_cur, edge_weight=weights[0])
                for w in weights[1:]:
                    m = torch.cat((m, self.propagate(edges, x=x_cur, edge_weight=w)), dim=1)
                x = self.multi_edges_rnn(m, x)
            else:
                # this method require num_layers equals len(weights)
                m = self.propagate(edges, x=x_cur, edge_weight=weights[i], size=None)
                x = self.rnn(m, x)
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                            self.out_channels,
                                            self.num_layers)

class AGMICellNeck(nn.Module):
    def __init__(self, in_channels=[6,8,16], out_channels=[8,16,32], kernel_size=[16,16,16], drop_rate=0.2, max_pool_size=[3,6,6], feat_dim=128):
        super().__init__()
        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(5376, feat_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x_cell_embed):
        x_cell_embed = self.cell_conv(x_cell_embed)

        x_cell_embed = x_cell_embed.view(-1, x_cell_embed.shape[1] * x_cell_embed.shape[2])
        
        # print(x_cell_embed.shape)

        x_cell_embed = self.fc(x_cell_embed)
        
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)                                      self.num_layers)

class AGMIFusionHead(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2, int(out_channels * 1.5)),
            nn.ReLU(),
            nn.Linear(int(out_channels * 1.5), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
        )

    def forward(self, x_cell_embed, drug_embed):
        out = torch.cat((x_cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)




"""
model = dict(
    type='AGMIDRPNet',
    drper=dict(
        type='AGMIDRPer',
        in_channel = 3,
        gather_width=6,
        drug_encoder=dict(
            type='DrugGATEncoder',
            num_features_xd=78, 
            heads=10, 
            output_dim=128, 
            gat_dropout=0.2
        ),
        genes_encoder=dict(
            type='MultiEdgeGatedGraphConv',
            out_channels=3, 
            num_layers=6, 
            num_edges=3,
            aggr='add',
            bias=True,
        ),
        head=dict(
            type='AGMIFusionHead',
            out_channels=128,
        ),
        neck=dict(
            type='AGMICellNeck',
            in_channels=[6,8,16], 
            out_channels=[8,16,32], 
            kernel_size=[16,16,16], 
            drop_rate=0.2, 
            max_pool_size=[3,6,6], 
            feat_dim=128
        ),
    ),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
)

train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE',
                         'R2', 'PEARSON', 'SPEARMAN'])

"""
