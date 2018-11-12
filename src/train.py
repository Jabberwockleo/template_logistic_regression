#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: train.py
Author: leowan(leowan)
Date: 2018/11/12 16:17:11
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

def main(dataset):
    """
        Tainer
    """
    # build graph
    tf.reset_default_graph()
    lr = logistic_regressor.LogisticRegressor(feature_num=conf.FEATURE_NUM,
        learning_rate=conf.LEARNING_RATE, random_seed=None)
    saver, logits, loss, train_op, stat_merged = lr.build_graph()
    
    # training
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # log dir
        log_dir = conf.LOG_DIR
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        # checkpoint dir
        chkpt_dir = conf.CHKPT_DIR
        if os.path.exists(chkpt_dir):
            shutil.rmtree(chkpt_dir)
        if not os.path.isdir(chkpt_dir):
            os.makedirs(chkpt_dir)

        for epoch in range(conf.EPOCHES):
            batch_cnt = 0
            batches_per_epoch = math.floor((len(dataset[0]) - 1) * 1.0 / conf.BATCH_SIZE) + 1
            best_loss = np.inf
            cur_loss = np.inf
            cur_accuracy = 0
            training_data = list(zip(dataset[0], dataset[1]))
            random.shuffle(training_data)
            for tu in utils.batch(training_data, n=conf.BATCH_SIZE):
                X, y = zip(*tu)
                y = np.expand_dims(y, 1)
                feed_dict = {
                    lr.input_feature_vectors: X,
                    lr.input_labels: y
                }
                sess.run(train_op, feed_dict=feed_dict)
                batch_cnt += 1
                global_step = epoch * batches_per_epoch + batch_cnt
                if global_step % conf.DISPLAY_STEP == 0:
                    in_f = dataset[0]
                    in_l = np.expand_dims(dataset[1], 1)
                    feed_dict = {
                        lr.input_feature_vectors: in_f,
                        lr.input_labels: in_l
                    }
                    cur_loss, cur_logits = sess.run([loss, logits], feed_dict=feed_dict)
                    summary_train = sess.run(stat_merged, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_train, global_step=global_step)
                    print("epoch: {}, global_step: {}, loss: {}, "
                        "accuracy: {}, f1: {}, auc: {}".format(
                        epoch, global_step, cur_loss,
                        metrics.calc_accuracy(cur_logits, in_l),
                        metrics.calc_f1(cur_logits, in_l),
                        metrics.calc_auc(cur_logits, in_l)))
            if cur_loss < best_loss:
                best_loss = cur_loss
                saver.save(sess, '{}/model'.format(chkpt_dir))

if __name__ == "__main__":
    # iris data
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
    
    main(dataset)
    
    # tensorboard
    # tensorboard --logdir /tmp/log/lr --port 8008