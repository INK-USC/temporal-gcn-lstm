import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, last=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.last = last

        """multiply src with edge data or not"""
        # self.msg_func = fn.copy_src(src='h', out='m')
        self.msg_func = fn.src_mul_edge(src='h', edge='w', out='m')

        self.reduce_func = fn.sum(msg='m', out='h')

    def apply(self, nodes):
        return {'h': F.relu(self.linear(nodes.data['h']))}

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.msg_func, self.reduce_func)
        g.apply_nodes(func=self.apply)
        if self.last:
            return dgl.mean_nodes(g, 'h')
        else:
            return g.ndata.pop('h')

    def cat(self, g):
        l = dgl.unbatch(g)
        return torch.stack([g.ndata['h'].view(-1) for g in l], 0)

    def max_pool(self, g):
        l = dgl.unbatch(g)
        return torch.stack([torch.max(g.ndata['h'], 0)[0] for g in l], 0)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size, False)
        self.gcn2 = GCNLayer(hidden_size, hidden_size, True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, inputs):

        h = self.gcn1(g, inputs)
        h = self.gcn2(g, h)

        return self.linear(h)
