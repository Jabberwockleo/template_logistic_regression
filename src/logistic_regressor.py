#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: logistic_regressor.py
Author: leowan(leowan)
Date: 2018/11/12 16:15:39
"""

import tensorflow as tf
import numpy as np

class LogisticRegressor(object):
    """
        Logistic Regressor
    """
    def __init__(self, feature_num, learning_rate=1e-2, random_seed=None):
        """
            Initializer
            Params:
                feature_num: feature number
                learning_rate: learning rate
                random_seed: random seed
        """
        self.feature_num = feature_num
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.construct_placeholders()
        
    def construct_placeholders(self):
        """
            Construct inpute placeholders
        """
        self.input_feature_vectors = tf.placeholder(shape=[None, self.feature_num],
            dtype=tf.float32)
        self.input_labels = tf.placeholder(shape=[None, 1],
            dtype=tf.float32)
    
    def build_graph(self):
        """
            Build graph
        """
        self.construct_weights()
        
        # network forward pass
        saver, logits = self.forward_pass()
        
        # loss function
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=self.input_labels))
        
        # training optimizer
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        
        # statistics
        tf.summary.scalar('loss', loss)
        stat_merged = tf.summary.merge_all()

        return saver, logits, loss, train_op, stat_merged
    
    def construct_weights(self):
        """
            Construct weights
        """
        self.weights = []
        self.biases = []
        
        for i in range(1):
            weight_key = "w_{}_{}".format(i, i+1)
            bias_key = "b_{}".format(i)
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[self.feature_num, 1],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[1],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # statistics
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])
            
    def forward_pass(self):
        """
            Forward pass
        """
        h = self.input_feature_vectors
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights) - 1:
                h = tf.nn.sigmoid(h)

        return tf.train.Saver(), h