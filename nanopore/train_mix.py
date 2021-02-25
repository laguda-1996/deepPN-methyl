import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.cuda.amp.autocast as autocast
#import torch.cuda.amp.autocast as GradScaler
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold
#import matplotlib.pyplot as plt
import argparse
from load_elec_seq import *
from mix_cnn_rnn import *
from inceptionv3_forseq import *
from tensorboardX import SummaryWriter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is {}'.format(device))

torch.backends.cudnn.enabled = False


def save_checkpoint(state,is_best,model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'checkpoint.pth.tar')

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')

def calculate_metric(gt, pred):
    confusion = sklearn.metrics.confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)
    # print('Sensitivity:', TP / float(TP + FN))
    # print('Specificity:', TN / float(TN + FP))





#in_csv = '../RNN/qian10W'



if __name__ == '__main__':

    torch.manual_seed(1000)

    parser = argparse.ArgumentParser()

    # main option
    parser.add_argument("-in_csv", "--positive_fasta", action="store", dest='in_csv', required=True,
                        help="positive fasta file")
    parser.add_argument("-neg_csv", "--negative_fasta", action="store", dest='neg_csv', required=True,
                        help="negative fasta file")

    parser.add_argument("-od", "--out_dir", action="store", dest='out_dir', required=True,
                        help="output directory")

    # rnn option
    parser.add_argument("-rnntype", "--rnn_type", action="store", dest='rnn_type', default='LSTM', type=str,
                        help="[capital] LSTM(default), GRU")
    parser.add_argument("-hidnum", "--hidden_num", action="store", dest='hidden_num', default=256, type=int,
                        help="rnn size")
    parser.add_argument("-rnndrop", "--rnn_drop", action="store", dest='rnn_drop', default=0.5, type=float,
                        help="rnn size")

    # fc option
    parser.add_argument("-fcdrop", "--fc_drop", action="store", dest='fc_drop', default=0.5, type=float,
                        help="Optional: 0.5(default), 0~0.5(recommend)")

    # optimization option
    parser.add_argument("-optim", "--optimization", action="store", dest='optim', default='Adam', type=str,
                        help="Optional: Adam(default), ")
    parser.add_argument("-epochs", "--max_epochs", action="store", dest='max_epochs', default=20, type=int,
                        help="max epochs")
    parser.add_argument("-lr", "--learning_rate", action="store", dest='learning_rate', default=0.001, type=float,
                        help="Adam: 0.0001(default), 0.0001~0.01(recommend)")
    # parser.add_argument("-lrstep", "--lr_decay_step", action="store", dest='lr_decay_step', default=10, type=int,
    #                     help="learning rate decay step")
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=256, type=int,
                        help="batch size")

    args = parser.parse_args()

    model_path = args.out_dir

    # in_csv = '../RNN/GATC_qian1W'
    # neg_in_csv = '../RNN/neg_qian1W'

    wordvec_len = 7

    HIDDEN_NUM = args.hidden_num

    LAYER_NUM = 3

    RNN_DROPOUT = args.rnn_drop
    FC_DROPOUT = args.fc_drop
    cell = args.rnn_type
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size





    tprs = []
    aucs = []
    ROC_aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    PR_aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)


    write = SummaryWriter('runs/exps')

    in_csv = args.in_csv
    neg_csv = args.neg_csv

    X, y = load_train_val(in_csv, neg_csv)
    #print(X.shape, y.shape)
    folds = StratifiedKFold(n_splits=5).split(X, y)
    for trained, valided in folds:
        X_train, y_train = X[trained], y[trained]
        X_test, y_test = X[valided], y[valided]
        X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test)
        X_train, y_train ,X_test = X_train.to(device), y_train.to(device), X_test.to(device)

        # print(X_train.shape)

        #checkpoint = torch.load('.' + '/' + 'checkpoint.pth.tar')

        # model = inception_v3(pretrained=False, num_classes=2, aux_logits=False, transform_input=False)
        model = ronghe_model(wordvec_len ,HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, cell)
        # model = deepsingalCNN_51(DROPOUT)
        # model.load_state_dict(checkpoint['state_dict'])

        model.to(device)

        loss = torch.nn.CrossEntropyLoss(reduction='sum')
        # loss = loss.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1.6677732579410078e-05)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1.2908389698714019e-05)
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3078372053802013)

        # write.add_graph(model, X_test)

        best_acc = 0
        best_train_accuracy_score = 0
        patience = 0


        def train(model, loss, optimizer, x, y):

            model.train()

            # Reset gradient
            optimizer.zero_grad()

            # Forward
            # with torch.cuda.amp.autocast():
            fx = model(x)
            loss = loss.forward(fx, y)

            pred_prob = F.log_softmax(fx, dim=1)

            # Backward
            loss.backward()
            # scaler.scale(loss).backward()

            # grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 14)
            # for p in model.parameters():
            #     p.data.add_(-LEARNING_RATE, p.grad.data)

            # Update parameters
            optimizer.step()
            # scaler.step(optimizer)

            # scaler.update()

            return loss.cpu().item(), pred_prob, list(np.array(y.cpu())), list(fx.data.cpu().detach().numpy().argmax(axis=1))  # cost,pred_probability and true y value


        def predict(model, x):
            model.eval()  # evaluation mode do not use drop out

            with torch.no_grad():
                fx = model(x)
            return fx


        EPOCH = args.max_epochs
        n_classes = 2
        n_examples = len(X_train)

        Loss_list = []
        Accuracy_list = []

        for i in range(EPOCH):
            start_time = time.time()

            cost = 0.
            y_pred_prob_train = []
            y_batch_train = []
            y_batch_pred_train = []

            num_batches = n_examples // BATCH_SIZE
            for k in range(num_batches):
                start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE

                # X_train, y_train, X_test, y_test = load_in_torch_fmt(X_train, y_train, X_test, y_test)
                # X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)

                # X_train[start:end] , y_train[start:end] = X_train[start:end].to(device), y_train[start:end].to(device)
                # print(X_train[start:end].device)

                output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end],
                                                                         y_train[start:end])
                cost += output_train

                # write.add_scalar('loss', output_train, global_step=k)
                # print(y_pred_prob.shape)

                prob_data = y_pred_prob.cpu().detach().numpy()
                # print(prob_data.shape)
                # if args.if_bce == 'Y':
                #     for m in range(len(prob_data)):
                #         y_pred_prob_train.append(prob_data[m][0])

                # else:
                for m in range(len(prob_data)):
                    # print(np.exp(prob_data)[m])
                    y_pred_prob_train.append(np.exp(prob_data)[m][1])

                # print(y_pred_prob_train)
                y_batch_train += y_batch
                y_batch_pred_train += y_pred_train

            scheduler.step()

            # rest samples
            # start, end = num_batches * BATCH_SIZE, n_examples
            # output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end],
            #                                                          y_train[start:end])
            # cost += output_train
            #
            # prob_data = y_pred_prob.cpu().detach().numpy()

            # for m in range(len(prob_data)):
            #     y_pred_prob_train.append(np.exp(prob_data)[m][1])
            # # print(y_pred_prob_train)
            #
            # y_batch_train += y_batch
            # y_batch_pred_train += y_pred_train

            # train AUC
            fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)


            train_accuracy_score = sklearn.metrics.accuracy_score(y_batch_train, y_batch_pred_train)

            train_specificity = calculate_metric(y_batch_train, y_batch_pred_train)

            train_recall_score = sklearn.metrics.recall_score(y_batch_train, y_batch_pred_train)
            train_precision_score = sklearn.metrics.precision_score(y_batch_train, y_batch_pred_train)
            train_f1_score = sklearn.metrics.f1_score(y_batch_train, y_batch_pred_train)
            train_mcc = sklearn.metrics.matthews_corrcoef(y_batch_train, y_batch_pred_train)

            # predict test
            fx_test = predict(model, X_test)
            y_pred_prob_test = []

            y_pred_test = fx_test.cpu().data.numpy().argmax(axis=1)
            prob_data = F.log_softmax(fx_test, dim=1).data.cpu().numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(np.exp(prob_data)[m][1])

            # test AUROC
            # fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test)
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)

            test_specificity = calculate_metric(y_test, y_pred_test)
            test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
            test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
            test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
            test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
            test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)

            end_time = time.time()
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)

            print("Epoch %d, cost = %f, AUROC_train = %0.6f, acc = %.6f%%, AUROC_test = %0.6f ,train_accuracy = %0.6f, train_recall = %0.6f, train_precision = %0.6f, train_f1_score = %0.6f, train_mcc = %0.6f, test_accuracy = %0.6f, test_recall = %0.6f, test_precision = %0.6f, test_sp = %0.6f, test_f1_score = %0.6f, test_mcc = %0.6f"
                % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), 100. * np.mean(y_pred_test == y_test),
                   auc(fpr_test, tpr_test), train_accuracy_score, train_recall_score, train_precision_score, train_f1_score,
                   train_mcc, test_accuracy_score, test_recall_score, test_precision_score, test_specificity, test_f1_score, test_mcc))

            # print(roc_auc_score(y_test, y_pred_test))
            print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            cur_acc = 100. * np.mean(y_pred_test == y_test)
            # cur_train_accuracy_score = train_accuracy_score
            # is_best = bool(cur_train_accuracy_score > best_train_accuracy_score)
            is_best = bool(cur_acc > best_acc)
            # best_train_accuracy_score = max(cur_train_accuracy_score, best_train_accuracy_score)
            best_acc = max(cur_acc, best_acc)
            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best, model_path)

            # patience

            # if auc(fpr_test, tpr_test) > 0.99:
            #     break

            if not is_best:
                patience += 1
                if patience >= 5:
                    break

            else:
                patience = 0

            if is_best:
                ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                                    model_path + '/' + 'predout_train.tsv')

                ytest_ypred_to_file(y_test, y_pred_prob_test,
                                    model_path + '/' + 'predout_val.tsv')

        fprArray.append(fpr_test)
        tprArray.append(tpr_test)
        thresholdsArray.append(thresholds_test)
        tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr_test, tpr_test)
        # print(roc_auc)
        ROC_aucs.append(roc_auc)

        recall_array.append(recall_test)
        precision_array.append(precision_test)
        precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
        pr_auc = auc(recall_test ,precision_test)
        PR_aucs.append(pr_auc)

    colors = cycle(['#5f0f40', '#9a031e', '#fb8b24', '#e36414', '#0f4c5c'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(fprArray)), colors):
        plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color,
                 label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ROC_mean_auc = auc(mean_fpr, mean_tpr)
    ROC_std_auc = np.std(ROC_aucs)
    # plt.plot(mean_fpr, mean_tpr, color='#ea7317',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (ROC_mean_auc, ROC_std_auc),
    #          lw=1.5, alpha=.9)
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(model_path + '/' + 'train_ROC.png')
    plt.show()
    plt.close(0)

    fig = plt.figure(1)
    for i, color in zip(range(len(recall_array)), colors):
        plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
                 label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    PR_mean_auc = auc(mean_recall, mean_precision)
    PR_std_auc = np.std(PR_aucs)

    # plt.plot(mean_recall, mean_precision, color='#ea7317',
    #          label=r'Mean PRC (AUPRC = %0.2f $\pm$ %0.2f)' % (PR_mean_auc, PR_std_auc),
    #          lw=1.5, alpha=.9)
    # std_precision = np.std(precisions, axis=0)
    # precision_upper = np.minimum(mean_precision + std_precision, 1)
    # precision_lower = np.maximum(mean_precision - std_precision, 0)
    # plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(model_path + '/' + 'train_pr.png')
    # plt.show()
    plt.close(0)

    print('> best acc:', best_acc)

    fig, ax = plt.subplots(1, 1)

    axins = ax.inset_axes((0.6, 0.4, 0.3, 0.3))

    for i, color in zip(range(len(fprArray)), colors):
        ax.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color,
                label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))
        axins.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color,
                   label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


    # 坐标轴的扩展比例（根据实际数据调整）

    # X轴的显示范围
    xlim0 = -0.01
    xlim1 = 0.01

    # Y轴的显示范围

    ylim0 = 0.99
    ylim1 = 1.01

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")

    # 画两条线
    xy = (xlim1, ylim1)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim0, ylim0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    plt.savefig(model_path + '/' + 'big_roc.png')
    plt.close(0)

    fig, ax = plt.subplots(1, 1)
    axins = ax.inset_axes((0.6, 0.4, 0.3, 0.3))

    for i, color in zip(range(len(recall_array)), colors):
        ax.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
                label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))
        axins.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
                   label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")

    # X轴的显示范围
    xlim0 = 0.99
    xlim1 = 1.01

    # Y轴的显示范围

    ylim0 = 0.99
    ylim1 = 1.01

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")

    # 画两条线
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    plt.savefig(model_path + '/' + 'big_pr.png')
    plt.close(0)

# for i in range(EPOCH):
#     start_time = time.time()
#
#     cost = 0.
#     y_pred_prob_train = []
#     y_batch_train = []
#
#     num_batches = n_examples // BATCH_SIZE
#     for k in range(num_batches):
#         start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
#         # X_train[start:end] , y_train[start:end] = X_train[start:end].to(device), y_train[start:end].to(device)
#         output_train ,y_pred_prob ,y_batch = train(model, loss, optimizer, X_train[start:end], y_train[start:end])
#         cost += output_train
#
#         prob_data = y_pred_prob.cpu().detach().numpy()
#         # if args.if_bce == 'Y':
#         #     for m in range(len(prob_data)):
#         #         y_pred_prob_train.append(prob_data[m][0])
#
#         # else:
#         for m in range(len(prob_data)):
#             y_pred_prob_train.append(np.exp(prob_data)[m][1])
#
#         y_batch_train += y_batch
#
#     scheduler.step()
#
#     #rest samples
#     #start, end = num_batches * BATCH_SIZE, n_examples
#     #output_train, y_pred_prob, y_batch = train(model, loss, optimizer, X_train[start:end], y_train[start:end])
#     #cost += output_train
#
#
#     #prob_data = y_pred_prob.cpu().detach().numpy()
#     # if args.if_bce == 'Y':
#     #     for m in range(len(prob_data)):
#     #         y_pred_prob_train.append(prob_data[m][0])
#
#
#     for m in range(len(prob_data)):
#         y_pred_prob_train.append(np.exp(prob_data)[m][1])
#
#
#     y_batch_train += y_batch
#
#     # train AUC
#     fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)
#
#     # predict test
#     output_test = predict(model, X_test)
#     y_pred_prob_test = []
#
#     # if args.if_bce == 'Y':
#     #     y_pred_test = []
#     #     prob_data = F.sigmoid(output_test).data.numpy()
#     #     for m in range(len(prob_data)):
#     #         y_pred_prob_test.append(prob_data[m][0])
#     #         if prob_data[m][0] >= 0.5:
#     #             y_pred_test.append(1)
#     #         else:
#     #             y_pred_test.append(0)
#     # else:
#     y_pred_test = output_test.data.cpu().numpy().argmax(axis=1)
#     # print(y_pred_test)
#     prob_data = F.log_softmax(output_test, dim=1).data.cpu().numpy()
#     # print(prob_data)
#     for m in range(len(prob_data)):
#         y_pred_prob_test.append(np.exp(prob_data)[m][1])
#
#     # test AUROC
#     fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
#     precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)
#
#
#     test_specificity = calculate_metric(y_test, y_pred_test)
#     test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
#     test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
#     test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
#     test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
#     test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)
#
#     end_time = time.time()
#     hours, rem = divmod(end_time - start_time, 3600)
#     minutes, seconds = divmod(rem, 60)
#
#     print("Epoch %d, cost = %f, AUROC_train = %0.3f, acc = %.2f%%, AUROC_test = %0.3f, test_accuracy = %0.3f, test_recall(sn) = %0.3f ,test_sp = %0.3f, test_precision = %0.3f, test_f1_score = %0.3f, test_mcc = %0.3f"
#           % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), 100. * np.mean(y_pred_test == y_test),
#              auc(fpr_test, tpr_test),  test_accuracy_score, test_recall_score, test_specificity, test_precision_score, test_f1_score, test_mcc))
#     Loss_list.append(cost / num_batches)
#     Accuracy_list.append(100. * np.mean(y_pred_test == y_test))
#     print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
#
#
#
#
#     cur_acc=100. * np.mean(y_pred_test == y_test)
#     is_best = bool(cur_acc >= best_acc)
#     best_acc = max(cur_acc, best_acc)
#     save_checkpoint({
#         'epoch': i+1,
#         'state_dict': model.state_dict(),
#         'best_accuracy': best_acc,
#         'optimizer': optimizer.state_dict()
#     }, is_best,model_path)
#
#
#
#     #patience
#     if not is_best:
#         patience+=1
#         if patience>= 5:
#             break
#
#     else:
#         patience=0
#
#
#
#
#     if is_best:
#         ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
#                             model_path + '/' + 'predout_train.tsv')
#
#         ytest_ypred_to_file(y_test, y_pred_prob_test,
#                             model_path + '/' + 'predout_val.tsv')
#
#
#
#     fprArray.append(fpr_test)
#     tprArray.append(tpr_test)
#     thresholdsArray.append(thresholds_test)
#     tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr_test, tpr_test)
#     ROC_aucs.append(roc_auc)
#
#     recall_array.append(recall_test)
#     precision_array.append(precision_test)
#     precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
#     pr_auc = auc(recall_test ,precision_test)
#     PR_aucs.append(pr_auc)
#
