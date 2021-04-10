import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnetwithCBAM import *


class deepsingalCNN_411(nn.Module):
    def __init__(self):
        super(deepsingalCNN_411, self).__init__()
        # [B, 1, 1, 411]
        self.basicconv0a = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 17), stride=(1,1), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )# [B, 32, 1, 399]
        self.basicconv0b = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 13), stride=(1,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )# [B, 64, 1, 194]
        self.maxpooling1b = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 64, 1, 97]

        self.basicconv0c = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 10), stride=(1,1)),  # [B, 128, 1, 88]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.maxpooling1c = nn.MaxPool2d(kernel_size=(1, 2)) # [B, 128, 1, 44]

        self.basicconv0d = torch.nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 9), stride=(1,1)) ,  # [B, 64, 1, 36]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpooling1d = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 64, 1, 18]

        self.basicconv0e = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 7), stride=(1, 1)),  # [B, 64, 1, 12]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.maxpooling1e = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 32, 1, 6]

        self.basicconv0f = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 1)),  # [B, 16, 1, 4]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.maxpooling1f = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 16, 1, 2]


    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.basicconv0a(x)
        #print(x.shape)
        x = self.basicconv0b(x)
        x = self.maxpooling1b(x)
        x = self.basicconv0c(x)
        x = self.maxpooling1c(x)
        x = self.basicconv0d(x)
        x = self.maxpooling1d(x)
        x = self.basicconv0e(x)
        x = self.maxpooling1e(x)
        x = self.basicconv0f(x)
        x = self.maxpooling1f(x)
        x = x.squeeze(2)
        elec_out = x.view(-1, 16*2)
        #print(elec_out.shape)

        return elec_out

class deepsingalRNN(nn.Module):
    def __init__(self, wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT, cell):
        super(deepsingalRNN ,self).__init__()
        self.rnn = torch.nn.Sequential()
        if cell == 'LSTM':
            self.rnn.add_module("lstm", nn.LSTM(input_size=wordvec_len, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                               bidirectional=True, dropout=DROPOUT))
        else:
            self.rnn.add_module("gru", nn.GRU(input_size=wordvec_len, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                                                    bidirectional=True, dropout=DROPOUT))


    def forward(self, x):
        # x > [batch_size, sequence_len, word_vec]
        x = x.permute(1,0,2)  # x > [sequence_len, batch_size, word_vec]
        out, _ = self.rnn(x) # out > [sequence_len, batch_size, num_directions*hidden_size]
        seq_out = torch.mean(out, 0)
        #print(seq_out.shape)

        return seq_out

class BiLSTM_Attention(nn.Module):
    def __init__(self, wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT):
        super(BiLSTM_Attention, self).__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=wordvec_len, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM, bidirectional=True, dropout=DROPOUT)
        self.fc1 = nn.Linear(HIDDEN_NUM * 2, 10)
        self.fc2 = nn.Linear(10, 2)
        # self.out = nn.Linear(HIDDEN_NUM * 2, num_classes)

    # lstm_output : [batch_size, n_step, HIDDEN_NUM * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, HIDDEN_NUM * 2, 3) # hidden : [batch_size, HIDDEN_NUM * num_directions(=2), 1(=n_layer)]
        hidden = torch.mean(hidden, 2).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, HIDDEN_NUM * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, HIDDEN_NUM * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, HIDDEN_NUM * num_directions(=2)]

    def forward(self, x):
        # input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = x.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        # hidden_state = Variable(torch.zeros(1*2, len(X), HIDDEN_NUM)) # [num_layers(=1) * num_directions(=2), batch_size, HIDDEN_NUM]
        # cell_state = Variable(torch.zeros(1*2, len(X), HIDDEN_NUM)) # [num_layers(=1) * num_directions(=2), batch_size, HIDDEN_NUM]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, HIDDEN_NUM]
        # output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, HIDDEN_NUM]
        attn_output = self.attention_net(output, final_hidden_state)
        return self.fc2(self.fc1(attn_output))# model : [batch_size, num_classes], attention : [batch_size, n_step]



class FC(nn.Module):
    def __init__(self, DROPOUT):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(1024, 2)
        # self.fc2 = nn.Linear(32, 2)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        # x = F.relu(x, inplace=True)
        # x = self.dropout(self.fc2(x))

        return x

class ronghe_model(nn.Module):
    def __init__(self, wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT, cell):
        super(ronghe_model, self).__init__()
        self.wordvec_len = wordvec_len
        self.HIDDEN_NUM = HIDDEN_NUM
        self.LAYER_NUM = LAYER_NUM
        self.DROPOUT = DROPOUT
        self.cell =  cell
        #self.cnn = deepsingalCNN_411()
        self.cnn = resnet18(pretrained=False, progress=False)
        self.rnn = deepsingalRNN(wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT, cell)
        #self.rnn = BiLSTM_Attention(wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT)
        self.fc = FC(DROPOUT)

    def forward(self, x):
        x1, x2 = x.split([17*7, 360], dim=2)
        
        #print(x1.squeeze(1).view(x.shape[0], 17, 7).shape, x2.unsqueeze(3).shape)
        #print(x1.squeeze(1).view(x.shape[0], 17, 7).shape)
        #print('+++++++++++++++++++')
        #print(x2.unsqueeze(3).shape)
        x_elec = self.cnn(x2.unsqueeze(3).permute(0,1,3,2))
        #print(x_elec.shape)
        x_seq = self.rnn(x1.squeeze(1).view(x.shape[0], 17, 7))
        #print(x_seq.shape)
        
        x = torch.cat([x_seq, x_elec], dim=1)
        x = self.fc(x)

        return x

