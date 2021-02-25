import numpy as np
import pandas as pd
import math
import argparse
# import tqdm
# import gpytorch
# from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.cuda.amp.autocast_mode as autocast
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
# from seq_load import *
# from model import *
from tensorboardX import SummaryWriter
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from model import CNN41_RNN


def train(model, parameters, x, y, device):
    class CNN41_RNN(nn.Module):
        def __init__(self, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT, FC_DROPOUT, CELL, conv1_outchannel, conv2_outchannel,
                     conv1_kernel, conv2_kernel):
            super(CNN41_RNN, self).__init__()
            # self.conv0 = torch.nn.Sequential(
            #     nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(inplace=True)
            # )  # [B, 32, 1, ]
            self.conv1 = torch.nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=conv1_outchannel, kernel_size=(1, conv1_kernel), stride=(1, 2), padding=(0, 2)),
                nn.BatchNorm2d(conv1_outchannel),
                nn.ReLU(inplace=True)
            )  # [B, 32, 1, ]
            self.conv2 = torch.nn.Sequential(
                nn.Conv2d(in_channels=conv1_outchannel, out_channels=conv2_outchannel, kernel_size=(1, conv2_kernel), stride=(1, 1)),
                nn.BatchNorm2d(conv2_outchannel),
                nn.ReLU(inplace=True)
            )  # [B, 64, 1, ]
            self.maxpooling1 = nn.MaxPool2d(kernel_size=(1, 2))  # [B, 64, 1, 9]

            self.rnn = torch.nn.Sequential()
            if CELL == 'LSTM':
                self.rnn.add_module("lstm", nn.LSTM(input_size=conv2_outchannel, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
                                                    bidirectional=True, dropout=RNN_DROPOUT))
            else:
                self.rnn.add_module("gru", nn.GRU(input_size=conv2_outchannel, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
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
            x = self.maxpooling1(x)  # x > [batch_size, channel(input_size), 1, seq_len]

            x = x.squeeze(2).permute(2, 0, 1)  # x > [sequence_len, batch_size, word_vec]

            out, _ = self.rnn(x)  # out > [sequence_len, batch_size, num_directions*hidden_size]
            out = torch.mean(out, 0)
            out = self.dropout(self.fc1(out))

            # out = F.relu(out)
            # out = self.fc2(out)
            return out

    # Initialize network

    # model = CNN41_RNN(parameters.get("hidnum", 128), parameters.get("layer", 3), RNN_DROPOUT=0.5, FC_DROPOUT=0.5, CELL='LSTM',
    #                   conv1_outchannel=parameters.get("conv1_outchannel", 64), conv2_outchannel=parameters.get("conv2_outchannel", 128),
    #                   conv1_kernel=parameters.get("conv1_kernel", 5) ,conv2_kernel=parameters.get("conv2_kernel", 3))
    model = model.to(device)
    # pyre-ignore [28]
    model.train()
    # Define loss and optimizer
    correct = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=parameters.get("lr", 0.001), weight_decay=parameters.get("weight_decay", 0.0))
    # criterion = nn.NLLLoss(reduction="sum")
    # optimizer = optim.SGD(
    #     net.parameters(),
    #     lr=parameters.get("lr", 0.001),
    #     momentum=parameters.get("momentum", 0.0),
    #     weight_decay=parameters.get("weight_decay", 0.0),
    # )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get("step_size", 10)),
        gamma=parameters.get("gamma", 0.5),  # default is no learning rate decay
    )
    # Train Network
    n_classes = 2
    n_examples = len(x)

    # pyre-fixme[6]: Expected `int` for 1st param but got `float`
    # move data to proper dtype and device
    for i in range(parameters.get('epoch', 30)):
        BATCH_SIZE  = parameters.get('batch', 64)
        num_batches = n_examples // BATCH_SIZE
        for k in range(num_batches):
            start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
            x_train_batch, y_train_batch = x[start:end], y[start:end]
        # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            fx = model(x_train_batch)
            loss = correct(fx, y_train_batch)

            pred_prob = F.log_softmax(fx, dim=1)

            # Backward
            loss.backward()
            # scaler.scale(loss).backward()

            # grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.get("_norm", 5))
            # for p in model.parameters():
            #     p.data.add_(-LEARNING_RATE, p.grad.data)

            # Update parameters
            optimizer.step()
            # scaler.step(optimizer)
        scheduler.step()

    return model

def evaluate(model, x, y) -> float:

    model.eval()

    with torch.no_grad():
        # move data to proper dtype and device
        # outputs = net(inputs)
        fx = model(x)
        # _ , predicted = torch.max(fx.data, 1)
        #
        # correct += (predicted == y).sum().item()

    # y_evaluate = fx.cpu().data.numpy().argmax(axis=1)
    _, y_pred_test= torch.max(fx.cpu().data, 1)
    y_test = y
    # acc = sklearn.metrics.accuracy_score(y, y_evaluate)
    # fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
    # precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)

    def calculate_metric(gt, pred):
        confusion = confusion_matrix(gt, pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        return TN / float(TN + FP)

    test_specificity = calculate_metric(y_test, y_pred_test)
    test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
    test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
    test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
    test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
    test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)

    print("test_accuracy = %0.3f, test_recall = %0.3f, test_precision = %0.3f, test_sp = %0.3f, test_f1_score = %0.3f, test_mcc = %0.3f"
        % (test_accuracy_score, test_recall_score, test_precision_score, test_specificity, test_f1_score, test_mcc))


    return (test_accuracy_score, test_recall_score, test_precision_score, test_specificity, test_f1_score, test_mcc)
