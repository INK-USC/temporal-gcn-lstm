import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm

"""Baseline activity sequence model"""


class LSTMs(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers, dropout=0):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        return last_hidden_out, self.linear(last_hidden_out)


class Data:
    def __init__(self, x, y, train_percent):
        # split into train, test index
        n = x.shape[0]

        split = int(n * train_percent)
        index = np.arange(n)
        # np.random.seed(42)
        np.random.shuffle(index)
        train_index, test_index = index[:split], index[split:]

        # load into pytorch Variable
        self.x_train = Variable(torch.FloatTensor(x[train_index, :, :]), requires_grad=False)
        self.x_test = Variable(torch.FloatTensor(x[test_index, :, :]), requires_grad=False)
        self.y_train = Variable(torch.LongTensor(y[train_index]), requires_grad=False)
        self.y_test = Variable(torch.LongTensor(y[test_index]), requires_grad=False)


def create_data(seq_length, df):
    seq_len = seq_length
    a = []

    """ fill in custom actions """
    # actions = []
    actions = ['action1', 'action2', 'action3']

    for user in df[actions].values:
        d = []
        for series in user:
            if series is np.nan:
                d.append([0.0 for _ in range(seq_len)])
            else:
                d.append(series[:seq_len])
        a.append(np.array(d).T)

    X = np.array(a)
    Y = df['label'].values.astype(int) + 1
    return X, Y


class Predictor(object):
    def __init__(self, params, data):
        self.params = params
        self.x_train = data.x_train
        self.y_train = data.y_train
        self.x_test = data.x_test
        self.y_test = data.y_test

    def train(self):
        self.model = LSTMs(self.params['input_dim'], self.params['hidden_dim'], self.params['out_dim'], self.params['layers'],
                           self.params['dropout'])
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss()
        #         self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

        best_test_f1 = 0
        best_train_f1 = 0

        for _ in tqdm(range(self.params['n_epoch']), ncols=100):

            last, out_vec = self.model(self.x_train)

            loss = self.criterion(out_vec, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_f1 = self.evaluate(self.x_train, self.y_train)
            test_f1 = self.evaluate(self.x_test, self.y_test)

            print(loss.item())
            print(train_f1, test_f1)

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_train_f1 = train_f1

        print('Best iteration train f1: ', best_train_f1)
        print('Best iteration test f1:  ', best_test_f1)

        return best_train_f1, best_test_f1

    def evaluate(self, x, y):
        # evaluate f1 score
        last, out = self.model(x)
        y_pred = nn.LogSoftmax(dim=1)(out)

        # multiclass f1 score, classes: [-1, 0, 1] //feed in (0, 1, 2) to satisfy function
        return f1_score(y, torch.argmax(y_pred, 1).data.numpy(), average='macro')

    def predict(self, x):
        x = Variable(torch.FloatTensor(x), requires_grad=False)
        last, out = self.model(x)
        return last, out


if __name__ == '__main__':
    """Input df as pandas dataframe of sequence data"""
    df = pd.DataFrame()
    X, Y = create_data(14, df)
    params = {'input_dim': 10, 'hidden_dim': 16, 'out_dim': 3, 'layers': 2,
              'dropout': 0.5, 'learning_rate': 0.01, 'n_epoch': 150}
    predictor = Predictor(params, Data(X, Y, 0.8))
    t1, t2 = predictor.train()
