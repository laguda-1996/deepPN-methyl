import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



class CNN_41(nn.Module):
    def __init__(self, DROPOUT):
        super(CNN_41, self).__init__()
        # [B, 1, 1, 41]
        self.basicconv0a = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 7), padding=(0,2), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )  # [B, 32, 1, 20]
        self.basicconv0b = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # [B, 64, 1, 16]
        self.maxpooling1b = nn.MaxPool2d(kernel_size=(1,2))# [B, 64, 1, 8]

        self.fc1 = nn.Linear(640, 2)
        # self.fc2 = nn.Linear(100, 2)


        self.dropout = nn.Dropout(DROPOUT)



    def forward(self, x):
        x = x.unsqueeze(3).permute(0, 2, 3, 1)
        x = self.basicconv0a(x)
        # x = self.basicconv0b(x)
        # x = self.maxpooling1b(x)
        x = x.squeeze(2)
        x = x.view(-1, 640)
        x = self.dropout(self.fc1(x))
        # x = F.relu(x)
        # x = self.dropout(self.fc2(x))
        return x


class RNN(nn.Module):
    def __init__(self, FC_DROPOUT, RNN_DROPOUT, CELL, HIDDEN_NUM, LAYER_NUM):
        super(RNN, self).__init__()

        self.rnn = torch.nn.Sequential()
        if CELL == 'LSTM':
            self.rnn.add_module("lstm", nn.LSTM(input_size=4, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                               bidirectional=True, dropout=RNN_DROPOUT))
        else:
            self.rnn.add_module("gru", nn.GRU(input_size=4, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                                                    bidirectional=True, dropout=RNN_DROPOUT))



        self.fc1 = nn.Linear(HIDDEN_NUM * 2, 2)
        # self.fc2 = nn.Linear(10, 2)

        self.dropout = nn.Dropout(FC_DROPOUT)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(1, 0, 2)
        # x = self.dropout(self.fc1(x))
        out, _ = self.rnn(x)  # out > [sequence_len, batch_size, num_directions*hidden_size]
        # print(out.shape)
        out = self.fc1(torch.mean(out, 0))

        return out


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = HIDDEN_NUM
        self.output_size = 11
        # self.dropout_p = dropout_p
        self.max_length = 11

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM, bidirectional=True, dropout=RNN_DROPOUT)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        # print(x.shape)
        output_prev, hidden = self.gru(x)
        # print(output_prev.shape)
        embedded = x.permute(1, 0, 2)
        # hidden = np.array(hidden)
        # hidden = torch.from_numpy(hidden)
        hidden = hidden.permute(1, 0, 2)
        # print(embedded[:,0].shape)
        # print(hidden[:,0].shape)

        # print(torch.mean(embedded,1).squeeze(1).shape)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[:,0], hidden[:,0]), 1)), dim=1)
        # print(torch.cat((embedded[:,0], hidden[:,0]), 1).shape)
        # print(self.attn(torch.cat((embedded[:,0], hidden[:,0]), 1)).shape)
        # print(attn_weights.shape)
        # print(attn_weights.unsqueeze(0).shape)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                          encoder_outputs.unsqueeze(0))

        # print(attn_weights.shape)
        # print(embedded.shape)
        attn_applied = torch.bmm(embedded.permute(0,2,1), attn_weights.unsqueeze(2))

        # print(attn_applied.shape)

        output = torch.cat((embedded[:,0], attn_applied.squeeze(2)), 1)
        # print(output.shape)
        output = self.attn_combine(output).unsqueeze(0)
        # print(output.shape)

        output = F.relu(output)
        output, _ = self.gru(output)
        # print(output.shape)

        # output = F.log_softmax(self.out(output[0]), dim=1)
        # return output, hidden, attn_weights
        return output.squeeze(0)

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)



class CNN41_RNN(nn.Module):
    def __init__(self , HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT, FC_DROPOUT, CELL):
        super(CNN41_RNN ,self).__init__()
        self.conv0 = torch.nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(1, 5), stride=(1,2), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
       )# [B, 32, 1, ]
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(1, 3), stride=(1,2), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )# [B, 32, 1, ]
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )# [B, 64, 1, ]
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 64, 1, 9]
        self.rnn_Attn = AttnDecoderRNN(128, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT)
        self.rnn = torch.nn.Sequential()
        if CELL == 'LSTM':
            self.rnn.add_module("lstm", nn.LSTM(input_size=128, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                               bidirectional=True, dropout=RNN_DROPOUT))
        else:
            self.rnn.add_module("gru", nn.GRU(input_size=64, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                                                    bidirectional=True, dropout=RNN_DROPOUT))

        self.fc1 = nn.Linear(HIDDEN_NUM * 2, 2)
        # self.fc2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(FC_DROPOUT)

    def forward(self, x):
        # x > [batch_size, sequence_len, word_vec]
        x = x.unsqueeze(3).permute(0, 2, 3, 1)
        # x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpooling1(x) # x > [batch_size, channel(input_size), 1, seq_len]
        # print(x.shape)
        x = x.squeeze(2).permute(2, 0, 1)  # x > [sequence_len, batch_size, word_vec]
        out, _  = self.rnn(x) # out > [sequence_len, batch_size, num_directions*hidden_size]
        out = torch.mean(out, 0)
        # out = self.rnn_Attn(x)
        out = self.dropout(self.fc1(out))
        # out = F.relu(out)
        # out = self.fc2(out)
        return out

