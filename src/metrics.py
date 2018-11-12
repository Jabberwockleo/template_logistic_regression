#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: metrics.py
Author: leowan(leowan)
Date: 2018/11/12 16:15:53
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def sigmoid(x):
    """
        Sigmoid
    """
    return 1 / (1 + np.exp(-x))

def calc_accuracy(logits, labels):
    """
        Calc accuracy
    """
    pred_labels = np.round(sigmoid(logits))
    match_score = np.equal(pred_labels, labels).astype(np.float32)
    return np.mean(match_score)

def calc_f1(logits, labels, log_confusion_matrix=False):
    """
        Calc F1 score
    """
    pred_labels = np.round(sigmoid(np.array(logits))).ravel()
    real_labels = np.array(labels).ravel()
    ind_1 = np.argwhere(real_labels == 1)
    ind_0 = np.argwhere(real_labels == 0)
    tp = np.sum(pred_labels[ind_1])
    tn = np.sum((1 - pred_labels)[ind_0])
    fp = np.sum(pred_labels[ind_0])
    fn = np.sum((1 - pred_labels)[ind_1])
    f1 = 2.0 * tp / (2*tp + fn + fp)
    acc = (tp + tn) * 1.0 / (tp + tn + fp + fn)
    if log_confusion_matrix is True:
        print("tp:{}, tn:{}, fp:{}, fn:{}, acc:{}, len:{}".format(
        tp, tn, fp, fn, acc, len(logits)))
    return f1

def calc_auc(logits, labels):
    """
        Calc AUC
    """
    return roc_auc_score(labels, logits)