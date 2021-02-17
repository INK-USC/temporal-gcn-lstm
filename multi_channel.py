import argparse

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm

from activity_seq_model import LSTMs, create_data
from gcn_model import GCN
from utils import nx_to_dgl

import pickle

"""Multi-channel temporal end-to-end framework"""

"""Input datasets
    df: dataframe with userid as index, contains labels, and activity sequence if needed
    macro: dataframe containing macroscopic data if needed
    graphs: dictionary with format {user_id: list of networkx graphs}}
    
    ** Modify acitivity and macro flags to run different versions of model to include features
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcn_in', type=int, default=2) # 12 in paper
    parser.add_argument('--gcn_hid', type=int, default=20)
    parser.add_argument('--gcn_out', type=int, default=20)
    parser.add_argument('--lstm_hid', type=int, default=32)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_drop', type=int, default=0)
    parser.add_argument('--a_in', type=int, default=3) # 10 in paper
    parser.add_argument('--macro_dim', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=100) # can increase
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--period', type=int, default=7)  # 14 days in paper

    parser.add_argument('--activity', type=bool, default=True)
    parser.add_argument('--macro', type=bool, default=True)

    parser.add_argument('--df_path', type=str)
    parser.add_argument('--macro_path', type=str)
    parser.add_argument('--graphs_path', type=str)

    args = parser.parse_args()

    # load data
    df = pd.read_pickle(args.df_path)
    if args.macro:
        macro = pd.read_pickle(args.macro_path).to_numpy()
    graphs_sep = pickle.load(open(args.graphs_path, 'rb'))  # {id: list of networkx graphs}

    # create graph input
    dgls, inputs, xav, eye, dim = [], [], [], [], 20
    for gid in tqdm(df.index):
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

    # train, test split
    n = len(dgls)
    split = int(n * .8)
    index = np.arange(n)
    np.random.seed(32)
    np.random.shuffle(index)
    train_index, test_index = index[:split], index[split:]

    # prep labels - +1 here bc original is [-1, 0, 1]
    train_labels, test_labels = Variable(torch.LongTensor((df['label'].values.astype(int) + 1)[train_index])), Variable(
        torch.LongTensor((df['label'].values.astype(int) + 1)[test_index]))

    # prep temporal graph data
    k = args.period
    trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]
    trainGs, testGs = [dgl.batch([u[i] for u in trainGs]) for i in range(k)], \
                      [dgl.batch([u[i] for u in testGs]) for i in range(k)]
    train_inputs, test_inputs = [inputs[i] for i in train_index], [inputs[i] for i in test_index]
    train_inputs, test_inputs = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs])) for i in range(k)],\
                                [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs])) for i in range(k)]
    train_xav, test_xav = [xav[i] for i in train_index], [xav[i] for i in test_index]
    train_xav, test_xav = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_xav])) for i in range(k)], [
        torch.FloatTensor(np.concatenate([inp[i] for inp in test_xav])) for i in range(k)]

    # prep activity sequence data
    if args.activity:
        X, Y = create_data(args.period, df)
        x_train = Variable(torch.FloatTensor(X[train_index, :, :]), requires_grad=False)
        x_test = Variable(torch.FloatTensor(X[test_index, :, :]), requires_grad=False)
        y_train = Variable(torch.LongTensor(Y[train_index]), requires_grad=False)
        y_test = Variable(torch.LongTensor(Y[test_index]), requires_grad=False)
    # prep macro data
    if args.macro:
        macro_train = Variable(torch.FloatTensor(macro[train_index]), requires_grad=False)
        macro_test = Variable(torch.FloatTensor(macro[test_index]), requires_grad=False)

    # define models
    model = LSTMs(args.gcn_out, args.lstm_hid, args.num_classes, args.lstm_layers, args.lstm_drop)
    net = GCN(args.gcn_in, args.gcn_hid, args.gcn_out)
    model1 = LSTMs(args.a_in, args.lstm_hid, args.num_classes, args.lstm_layers, args.lstm_drop)
    linear_in_dim = args.num_classes + (args.num_classes if args.activity else 0) + (args.macro_dim if args.macro else 0)
    linear = nn.Linear(linear_in_dim, args.num_classes)

    parameters = list(net.parameters()) + list(model.parameters()) + list(model1.parameters()) + list(linear.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    dropout = nn.Dropout(args.drop)

    for epoch in tqdm(range(args.epoch)):

        # train
        model.train()
        net.train()

        # Run through GCN
        sequence = torch.stack([net(trainGs[i], train_inputs[i]) for i in range(k)], 1)
        # Temporal graph embeddings through lstm
        last, out = model(sequence)

        cat = out
        # Activity sequence through lstm
        if args.activity:
            last1, out1 = model1(x_train)
            cat = torch.cat((cat, out1), 1)
        if args.macro:
            cat = torch.cat((cat, macro_train), 1)
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
        cat = out
        if args.activity:
            last1, out1 = model1(x_test)
            cat = torch.cat((cat, out1), 1)
        if args.macro:
            cat = torch.cat((cat, macro_test), 1)
        mapped = linear(cat)

        test_logp = F.log_softmax(mapped, 1)
        test_f1 = f1_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Train Loss: %.4f | Train F1: %.4f | Test F1: %.4f' % (epoch, loss.item(), f1, test_f1))

