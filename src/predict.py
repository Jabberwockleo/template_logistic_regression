#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: predict.py
Author: leowan(leowan)
Date: 2018/11/12 16:18:38
"""

import os
import math
import random
import shutil

import numpy as np
import tensorflow as tf
import config as conf
import logistic_regressor
import metrics as metrics
import utils as utils

def main_test(dataset):
    """
        Predictor Test
    """
    tf.reset_default_graph()
    lr = logistic_regressor.LogisticRegressor(feature_num=conf.FEATURE_NUM,
        learning_rate=conf.LEARNING_RATE, random_seed=None)
    saver, logits, loss, train_op, stat_merged = lr.build_graph()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, '{}/model'.format(conf.CHKPT_DIR))

        in_feature_vecs = dataset[0][:100]
        in_labels = dataset[1][:100]
        in_labels = np.expand_dims(in_labels, 1)
        feed_dict = {
            lr.input_feature_vectors: in_feature_vecs,
            lr.input_labels: in_labels
        }
        out_logits, out_weights, out_biases = sess.run(
            [logits, lr.weights, lr.biases], feed_dict=feed_dict)
        print("accuracy: {}, f1: {}, auc: {}".format(
            metrics.calc_accuracy(out_logits, in_labels),
            metrics.calc_f1(out_logits, in_labels, log_confusion_matrix=True),
            metrics.calc_auc(out_logits, in_labels)))
        print("weights: ", out_weights)
        print("biases: ", out_biases)


def main(dataset):
    """
        Predictor
    """
    tf.reset_default_graph()
    lr = logistic_regressor.LogisticRegressor(feature_num=conf.FEATURE_NUM,
        learning_rate=conf.LEARNING_RATE, random_seed=None)
    lr.construct_placeholders()
    lr.construct_weights()
    saver, logits = lr.forward_pass()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, '{}/model'.format(conf.CHKPT_DIR))

        in_feature_vecs = dataset[0][:100]
        feed_dict = {
            lr.input_feature_vectors: in_feature_vecs
        }
        out_logits = sess.run([logits], feed_dict=feed_dict)
        print("logits: {}".format(out_logits))


if __name__ == "__main__":
    import numpy as np
    from sklearn import datasets
    dataset_ori = datasets.load_iris(return_X_y=True)
    y_label = map(lambda x: x == 0, dataset_ori[1])
    dataset = []
    dataset.append(dataset_ori[0])
    dataset.append(np.array(list(y_label)).astype(int))

    # mock data
    #def mock_boundary_func(X):
    #    # mock weights is (1, 2, 3, ..)
    #    return np.sum(np.dot(X, list(range(1, len(X) + 1))))
    #dataset = []
    #dataset.append(np.random.standard_normal(size=(1000, FEATURE_NUM)))
    #dataset.append(np.array(list(map(lambda x: mock_boundary_func(x) >= 0, dataset[0]))).astype(int))

    pred.main_test(dataset)
        