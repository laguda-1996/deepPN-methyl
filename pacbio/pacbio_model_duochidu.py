import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import math



class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = HIDDEN_NUM
        self.output_size = 9
        # self.dropout_p = dropout_p
        self.max_length = 9

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

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class CNN41_RNN(nn.Module):
    def __init__(self , HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT, FC_DROPOUT, CELL, o1=256, o2=256, o3=256):
        super(CNN41_RNN ,self).__init__()
        self.conv0 = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=o1, kernel_size=(1, 5), stride=(1,1), padding=(0,2)),
            nn.BatchNorm2d(o1),
            nn.ReLU(inplace=True)
        ) # [B, 32, 1, ]
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=o2, kernel_size=(1, 3), stride=(1,1), padding=(0,1)),
            nn.BatchNorm2d(o2),
            nn.ReLU(inplace=True)
        )# [B, 32, 1, ]
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=o3, kernel_size=(1, 1), stride=(1,1)),
            nn.BatchNorm2d(o3),
            nn.ReLU(inplace=True)
        )# [B, 64, 1, ]
        #
        self.maxpooling01 = nn.MaxPool2d(kernel_size=(1, 1))  # [B, 64, 1, 9]
        self.convp01 = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=o3, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(o3),
            nn.ReLU(inplace=True)
        )  # [B, 64, 1, ]





        self.conv00 = torch.nn.Sequential(
            nn.Conv2d(in_channels=o1+o2+o3, out_channels=o1, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(o1),
            nn.ReLU(inplace=True)
        )  # [B, 32, 1, ]
        self.conv11 = torch.nn.Sequential(
            nn.Conv2d(in_channels=o1+o2+o3, out_channels=o2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(o2),
            nn.ReLU(inplace=True)
        )  # [B, 32, 1, ]
        self.conv22 = torch.nn.Sequential(
            nn.Conv2d(in_channels=o1+o2+o3, out_channels=o3, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(o3),
            nn.ReLU(inplace=True)
        )
        # #
        # self.maxpooling02 = nn.MaxPool2d(kernel_size=(1, 1))  # [B, 64, 1, 9]
        # self.convp02 = torch.nn.Sequential(
        #     nn.Conv2d(in_channels=o1+o2+o3+o3, out_channels=o3, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(o3),
        #     nn.ReLU(inplace=True)
        # )  # [B, 64, 1, ]



        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 64, 1, 9]
        # self.rnn_Attn = AttnDecoderRNN(256, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT)
        self.transformr = transmformer()
        self.rnn = torch.nn.Sequential()
        if CELL == 'LSTM':
            self.rnn.add_module("lstm", nn.LSTM(input_size=o1+o2+o3+o3, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                               bidirectional=True, dropout=RNN_DROPOUT))
        else:
            self.rnn.add_module("gru", nn.GRU(input_size=o3, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                                                    bidirectional=True, dropout=RNN_DROPOUT))

        self.fc1 = nn.Linear(HIDDEN_NUM * 2, 2)
        # self.fc1 = nn.Linear(9*128, 2)
        self.dropout = nn.Dropout(FC_DROPOUT)

    def forward(self, x):
        # x > [batch_size, sequence_len, word_vec]
        x = x.unsqueeze(3).permute(0, 2, 3, 1)
        # x = self.conv0(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        x1 = self.conv0(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)
        x4 = self.maxpooling01(x)
        x4 = self.convp01(x4)

        x_1 = torch.cat((x1, x2, x3, x4), axis=1)
        # print(x_1.shape)

        # x11 = self.conv00(x_1)
        # x12 = self.conv11(x_1)
        # x13 = self.conv22(x_1)
        # # x14 = self.maxpooling02(x_1)
        # # x14 = self.convp02(x14)
        #
        # x_2 = torch.cat((x11, x12, x13), axis=1)
        #
        # x21 = self.conv00(x_2)
        # x22 = self.conv11(x_2)
        # x23 = self.conv22(x_2)
        #
        # x = torch.cat((x21, x22, x23), axis=1)







        x = self.maxpooling1(x_1) # x > [batch_size, channel(input_size), 1, seq_len]
        x = x.squeeze(2).permute(2, 0, 1)  # x > [sequence_len, batch_size, word_vec]

        # x = x.permute(1, 0, 2)
        # print(x.shape)
        # out = self.transformr(x)


        out, _  = self.rnn(x) # out > [sequence_len, batch_size, num_directions*hidden_size]

        out = torch.mean(out, 0)
        # print(out.shape)

        # out = self.rnn_Attn(x)
        out = self.dropout(self.fc1(out))
        # out = F.relu(out)
        # out = self.fc2(out)
        return out


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.5, max_len=8):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        # print(x.size()[:2])
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )


class transmformer(nn.Module):
    def __init__(self, embed_dim = 128, seqlen = 9):
        super(transmformer,self).__init__()
        self.embed_dim = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2)

        # self.pos_encoder = PositionalEncoding(8, dropout=0.1)

        self.pos_encoder = PositionEmbedding(seqlen, embed_dim)
        self.transformer_encoder123 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # self.fc = nn.Linear(17*4, 2)


    def forward(self, x):
        # x = x.permute(1, 0, 2)
        embed_dim = 128
        seqlen = 9

        x = x* math.sqrt(embed_dim)
        # print(x.shape)
        x = self.pos_encoder(x)
        # print(x.shape)
        x = self.transformer_encoder123(x)
        # print(x.shape)
        x = x.view(-1, seqlen*embed_dim)
        # print(x.shape)

        # out = self.fc(x)

        return x





# class CNN_41(nn.Module):
#     def __init__(self, DROPOUT):
#         super(CNN_41, self).__init__()
#         # [B, 1, 1, 41]
#         self.basicconv0a = torch.nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 7), padding=(0,2), stride=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )  # [B, 32, 1, 20]
#         self.basicconv0b = torch.nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5)),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )  # [B, 64, 1, 16]
#         self.maxpooling1b = nn.MaxPool2d(kernel_size=(1,2))# [B, 64, 1, 8]
#
#         self.fc1 = nn.Linear(640, 2)
#         # self.fc2 = nn.Linear(100, 2)
#
#
#         self.dropout = nn.Dropout(DROPOUT)
#
#
#
#     def forward(self, x):
#         x = x.unsqueeze(3).permute(0, 2, 3, 1)
#         x = self.basicconv0a(x)
#         # x = self.basicconv0b(x)
#         # x = self.maxpooling1b(x)
#         x = x.squeeze(2)
#         x = x.view(-1, 640)
#         x = self.dropout(self.fc1(x))
#         # x = F.relu(x)
#         # x = self.dropout(self.fc2(x))
#         return x
#
#
# class RNN(nn.Module):
#     def __init__(self, FC_DROPOUT, RNN_DROPOUT, CELL, HIDDEN_NUM, LAYER_NUM):
#         super(RNN, self).__init__()
#
#         self.rnn = torch.nn.Sequential()
#         if CELL == 'LSTM':
#             self.rnn.add_module("lstm", nn.LSTM(input_size=4, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
#                                bidirectional=True, dropout=RNN_DROPOUT))
#         else:
#             self.rnn.add_module("gru", nn.GRU(input_size=4, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
#                                                     bidirectional=True, dropout=RNN_DROPOUT))
#
#
#
#         self.fc1 = nn.Linear(HIDDEN_NUM * 2, 2)
#         # self.fc2 = nn.Linear(10, 2)
#
#         self.dropout = nn.Dropout(FC_DROPOUT)
#
#     def forward(self, x):
#         # print(x.shape)
#         x = x.permute(1, 0, 2)
#         # x = self.dropout(self.fc1(x))
#         out, _ = self.rnn(x)  # out > [sequence_len, batch_size, num_directions*hidden_size]
#         # print(out.shape)
#         out = self.fc1(torch.mean(out, 0))
#
#         return out