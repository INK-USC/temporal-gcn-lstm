import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm_notebook

"""Multi-channel temporal end-to-end framework

Static GCN, Temporal GCN models can be acheived by modifying data loading and training process
"""


class LSTMs(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout=0):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, 3)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        return last_hidden_out, self.linear(last_hidden_out)


def create_data(seq_length, df):
    seq_len = seq_length
    a = []
    """fill in custom actions"""
    actions = []
    for user in df[actions].values:
        d = []
        for series in user:
            if series is np.nan:
                d.append([0.0 for _ in range(seq_len)])
            else:
                d.append(series[:seq_len])
        a.append(np.array(d).T)

    X = np.array(a)
    Y = df['label'].values + 1
    return X, Y


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


def nx_to_dgl(g):
    G = dgl.DGLGraph()
    d = {n: i for i, n in enumerate(list(g.nodes()))}
    G.add_nodes(g.number_of_nodes())
    for e in list(g.edges()):
        G.add_edge(d[e[0]], d[e[1]], {'w': torch.FloatTensor([[g[e[0]][e[1]]['weight']]])})
    return G


if __name__ == '__main__':

    """load custom data"""
    df = pd.DataFrame()
    macro = pd.DataFrame()
    graphs_sep = {}       # {id: networkx graph}

    dgls, inputs, xav, eye, dim = [], [], [], [], 20
    for gid in tqdm_notebook(df.index):
        g_list = graphs_sep[gid]
        temp_g, temp_adj, temp_xav = [], [], []
        for g in g_list:
            G = nx_to_dgl(g)
            temp_g.append(G)
            temp_adj.append(np.array(nx.adj_matrix(g).todense()))
            temp_xav.append(nn.init.xavier_uniform_(torch.zeros([12, dim])))
        dgls.append(temp_g)
        inputs.append(temp_adj)
        xav.append(temp_xav)

    X, Y = create_data(14, df)

    n = len(dgls)
    split = int(n * .8)
    index = np.arange(n)
    np.random.seed(9835)
    np.random.shuffle(index)
    train_index, test_index = index[:split], index[split:]

    k = 14

    trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]
    trainGs, testGs = [dgl.batch([u[i] for u in trainGs]) for i in range(k)], [dgl.batch([u[i] for u in testGs]) for i in
                                                                               range(k)]
    train_labels, test_labels = Variable(torch.LongTensor((df['label'].values + 1)[train_index])), Variable(
        torch.LongTensor((df['label'].values + 1)[test_index]))

    # graph data
    train_inputs, test_inputs = [inputs[i] for i in train_index], [inputs[i] for i in test_index]
    train_inputs, test_inputs = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs])) for i in range(k)], [
        torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs])) for i in range(k)]
    train_xav, test_xav = [xav[i] for i in train_index], [xav[i] for i in test_index]
    train_xav, test_xav = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_xav])) for i in range(k)], [
        torch.FloatTensor(np.concatenate([inp[i] for inp in test_xav])) for i in range(k)]

    # activity sequence data
    x_train = Variable(torch.FloatTensor(X[train_index, :, :]), requires_grad=False)
    x_test = Variable(torch.FloatTensor(X[test_index, :, :]), requires_grad=False)
    y_train = Variable(torch.LongTensor(Y[train_index]), requires_grad=False)
    y_test = Variable(torch.LongTensor(Y[test_index]), requires_grad=False)
    macro_train = Variable(torch.FloatTensor(macro[train_index]), requires_grad=False)
    macro_test = Variable(torch.FloatTensor(macro[test_index]), requires_grad=False)

    # define models
    model = LSTMs(20, 32, 2, 0)
    net = GCN(12, 20, 20)
    model1 = LSTMs(10, 32, 2, 0)
    linear = nn.Linear(10, 3)

    parameters = list(net.parameters()) + list(model.parameters()) + list(model1.parameters()) \
        # + list(linear.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    optimizer2 = torch.optim.Adam(linear.parameters(), lr=0.01, weight_decay=0.001)
    dropout = nn.Dropout(0.5)

    for epoch in tqdm_notebook(range(1000)):

        #train
        model.train()
        net.train()

        # Run through GCN
        sequence = torch.stack([net(trainGs[i], train_inputs[i]) for i in range(k)], 1)
        # Temporal graph embeddings through lstm
        last, out = model(sequence)

        # Activity sequence through lstm
        last1, out1 = model1(x_train)

        cat = torch.cat((out, out1, macro_train), 1)
        cat = dropout(cat)

        mapped = linear(cat)

        logp = F.log_softmax(mapped, 1)
        loss = F.nll_loss(logp, train_labels)

        f1 = f1_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')

        # eval
        model.eval()
        net.eval

        test_sequence = torch.stack([net(testGs[i], test_inputs[i]) for i in range(k)], 1)
        last, out = model(test_sequence)
        last1, out1 = model1(x_test)

        cat = torch.cat((out, out1, macro_test), 1)
        mapped = linear(cat)

        test_logp = F.log_softmax(mapped, 1)
        test_f1 = f1_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')

        # back propagation
        optimizer.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer.step()
        optimizer2.step()

        print('Epoch %d | Train Loss: %.4f | Train F1: %.4f | Test F1: %.4f' % (epoch, loss.item(), f1, test_f1))

