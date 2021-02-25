import numpy as np
import sys
import pandas as pd
import random

import torch
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools

def neg_data(kmer, in_csv):
    df = pd.read_csv(in_csv,header=None)
    num = len(df)
    sequence = []
    for i in range(num):
        a = ''.join(random.choice('ATCG') for _ in range(kmer))
        # print(a[0],np.random.rand(-1,1))
        a = convert_seq_to_bicoding(a)
        sequence.append(a)
    return sequence

def convert_seq_to_bicoding(seq):
    #return bicoding for a sequence
    seq=seq.replace('U','T') #turn rna seq to dna seq if have
    feat_bicoding=[]
    bicoding_dict={'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    for each_nt in seq:
        feat_bicoding+=bicoding_dict[each_nt]
    return feat_bicoding


def load_data(in_csv):
    df = pd.read_csv(in_csv, header=None, sep='\t')
    seq = []
    for i in range(len(df)):
        b = df.iloc[i, 6]
        b = np.array(convert_seq_to_bicoding(b))
        singal_mean = pd.Series(df.iloc[i, 7].split(',')).apply(lambda x: float(x))
        singal_std = pd.Series(df.iloc[i, 8].split(',')).apply(lambda x: float(x))
        singal_len = pd.Series(df.iloc[i, 9].split(',')).apply(lambda x: float(x))
        c = zip(list(singal_mean), list(singal_std), list(singal_len))
        c = np.array(list(c))
        m = b.reshape(17, 4)
        n = np.concatenate((m, c), axis=1)
        n = n.flatten()
        seq.append(n.tolist())

    elec = []
    for i in range(len(df)):
        a = df.iloc[i, 7].split(',')
        b = df.iloc[i, 8].split(',')
        c = df.iloc[i, 9].split(',')
        d = df.iloc[i, 10].split(',')
        #quan = a+b+c+d[0:248]
        quan = d
        elec.append(quan)

    #new = pd.DataFrame(elec)
    #new.fillna(0)
    #t = new.applymap(lambda x: float(x))
    #v = np.array(t)
    v = np.array(elec).astype('float')

    return seq, v, v.shape[1], len(df)

def load_train_val(in_csv, neg_in_csv):
    # def suijishu(lie, singal_num):
    #     return np.random.rand(lie, singal_num)

    pos = load_data(in_csv)
    neg = load_data(neg_in_csv)
    # data_neg_train = suijishu(p[2], p[1]).tolist()
    # print(type(data_pos_train))

    per_elec_seq_pos = list(zip(np.array(pos[0]), np.array(pos[1].tolist())))
    per_elec_seq_neg = list(zip(np.array(neg[0]), np.array(neg[1].tolist())))

    data_train = np.array([list(_) + [1] for _ in per_elec_seq_pos] + [list(_) + [0] for _ in per_elec_seq_neg])

    np.random.seed(42)
    np.random.shuffle(data_train)

    X = np.array([list(_[0]) + list(_[1]) for _ in data_train])
    y = np.array([_[-1] for _ in data_train])
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/8, random_state=42)
    #
    # return X_train, X_test, y_train, y_test

    return X, y


def load_in_torch_fmt(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], 1, 17*7+360)
    X_test = X_test.reshape(X_test.shape[0], 1, 17*7+360)
    #print(X_train.shape)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    #y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()

    return X_train, y_train, X_test, y_test





