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



def train(model, parameters, x, y, device):

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
    for i in range(parameters.get('epoch', 20)):
        BATCH_SIZE  = parameters.get('batch', 256)
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
