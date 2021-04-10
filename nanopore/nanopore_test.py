import numpy as np
import pandas as pd
import math
#import tqdm
#import gpytorch
# from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from load_elec_seq import *
from mix_cnn_rnn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metric(true, pred):
    confusion = confusion_matrix(true, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)


wordvec_len = 7
HIDDEN_NUM = 256
LAYER_NUM = 3
DROPOUT = 0.5
LEARNING_RATE = 0.001
cell = 'LSTM'


def predict(model, x):
    model.eval() #evaluation mode do not use drop out
    fx = model(x)
    return fx.cpu()


tprs = []
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


#pos_train_fa = './RMVar数据库/cd80%_RMVar+-.fasta'
#neg_train_fa = './RMVar数据库/neg_51bp.fasta'

#pos_train_fa = './transcirc数据库测试/shuffle_motif.fasta'
#neg_train_fa = './transcirc数据库测试/neg_80Wcdhit.fasta'


# in_csv = 'GATC_hou5K'
# neg_in_csv = 'C_hou5K'

in_csv = './测试数据/GATC_6K'
neg_in_csv = './测试数据/neg_6K'



model_path = '.'
if model_path[-1] == '/':
    model_path = model_path[:-1]
# checkpoint = torch.load(model_path + '/' + 'checkpoint_by_RMVarposandneg.pth.tar', map_location=torch.device('cpu'))
#checkpoint = torch.load(model_path + '/' + 'checkpoint_by_RMVarposandneg.pth.tar')
checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar')



model = ronghe_model(wordvec_len ,HIDDEN_NUM, LAYER_NUM, DROPOUT, cell)
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])

# data_pos_train = load_data_bicoding(pos_train_fa)
# data_neg_train = load_data_bicoding(neg_train_fa)
#
# data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
# np.random.seed(42)
# np.random.shuffle(data_train)
#
# X_test = np.array([_[:-1] for _ in data_train])
# y_test = np.array([_[-1] for _ in data_train])

pos = load_data(in_csv)
neg = load_data(neg_in_csv)

per_elec_seq_pos = list(zip(np.array(pos[0]), np.array(pos[1].tolist())))
per_elec_seq_neg = list(zip(np.array(neg[0]), np.array(neg[1].tolist())))

data_train = np.array([list(_) + [1] for _ in per_elec_seq_pos] + [list(_) + [0] for _ in per_elec_seq_neg])

np.random.seed(42)
np.random.shuffle(data_train)

X_test = np.array([list(_[0]) + list(_[1]) for _ in data_train])
y_test = np.array([_[-1] for _ in data_train])

X_test = X_test.reshape(X_test.shape[0], 1, 17 * 7 + 360)
X_test = torch.from_numpy(X_test).float()
X_test = X_test.to(device)


batch_size = 256
i = 0
N = X_test.shape[0]
y_pred_test = []
y_pred_prob_test = []

while i + batch_size < N:
    x_batch = X_test[i:i + batch_size]

    fx = predict(model, x_batch)

    y_pred = fx.cpu().data.numpy().argmax(axis=1)

    prob_data = F.log_softmax(fx.cpu(), dim=1).data.numpy()
    for m in range(len(prob_data)):
        y_pred_prob_test.append(np.exp(prob_data)[m][1])

    y_pred_test += list(y_pred)

    i += batch_size

x_batch = X_test[i:N]
fx = predict(model, x_batch)
y_pred = fx.cpu().data.numpy().argmax(axis=1)
prob_data = F.log_softmax(fx, dim=1).cpu().data.numpy()
for m in range(len(prob_data)):
    y_pred_prob_test.append(np.exp(prob_data)[m][1])

y_pred_test += list(y_pred)


# test metrics
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)

confusion = confusion_matrix(y_test, y_pred_test)

test_specificity = calculate_metric(y_test, y_pred_test)
# test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)

print(" acc = %.3f%%, AUROC_test = %0.3f, test_recall(sn) = %0.4f ,test_sp = %0.4f, test_precision = %0.4f, test_f1_score = %0.4f, test_mcc = %0.4f"% (100. * np.mean(y_pred_test == y_test), auc(fpr_test, tpr_test), test_recall_score, test_specificity, test_precision_score, test_f1_score,
       test_mcc))

fprArray.append(fpr_test)
tprArray.append(tpr_test)
thresholdsArray.append(thresholds_test)
tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
tprs[-1][0] = 0.0
roc_auc = auc(fpr_test, tpr_test)
ROC_aucs.append(roc_auc)

recall_array.append(recall_test)
precision_array.append(precision_test)
precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
pr_auc = auc(recall_test, precision_test)
PR_aucs.append(pr_auc)



# classes = list(set(y_test))
fig = plt.figure(0)
classes = ['nonm6A', 'm6A']
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('pred')
plt.ylabel('fact')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])

plt.savefig(model_path + '/' + 'test_hunxiao.png')
plt.close(0)

colors = cycle(['#5f0f40', '#9a031e' ,'#fb8b24', '#e36414', '#0f4c5c', '#4361ee', '#c44536', '#bdb2ff'])
## ROC plot for CV
fig = plt.figure(0)
for i, color in zip(range(len(fprArray)), colors):
    plt.plot(fprArray[i], tprArray[i], lw=1.5, alpha=0.9, color='r',
             label=' (AUC = %0.4f)' % (ROC_aucs[i]))
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#c4c7ff', alpha=.8)
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# ROC_mean_auc = auc(mean_fpr, mean_tpr)
# ROC_std_auc = np.std(ROC_aucs)
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

plt.savefig(model_path + '/' + 'test_ROC.png')
# plt.close(0)

fig = plt.figure(1)
for i, color in zip(range(len(recall_array)), colors):
    plt.plot(recall_array[i], precision_array[i], lw=1.5, alpha=0.9, color='r',
             label=' (AUPRC = %0.4f)' % ( PR_aucs[i]))
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

plt.savefig(model_path + '/' + 'test_pr.png')
plt.close(0)





