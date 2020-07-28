#!/usr/bin/python
# -*- coding: utf-8 -*-
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Author: Zhongsheng Chen
# Date: 05/09/2020
# Copyright: Copyright 2020, Beijing University of Chemical Technology
# License: The MIT License (MIT)
# Email: zschen@mail.buct.edu.cn
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

"""multilayer perceptron.

Implement of MLP. The architecture of MLP is defined by create_network()

"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.preprocessing import StandardScaler

import keras

x_scaler = StandardScaler()
y_scaler = StandardScaler()


class MLPRegressor(keras.Model):
    def __init__(self, param_args):
        super(MLPRegressor, self).__init__()

        self.input_size = param_args.input_size
        self.output_size = param_args.output_size

        self.lr = param_args.lr
        self.decay = param_args.decay
        self.batch_size = param_args.batch_size
        self.epochs = param_args.epochs
        self.dropout_rate = param_args.dropout_rate
        self.activation = param_args.activation

        if param_args.optimizer == "Adam":
            self.optimizer = optimizers.Adam(lr=self.lr, decay=self.decay)
        elif param_args.optimizer == "SDG":
            self.optimizer = optimizers.SGD(lr=self.lr, decay=self.decay)
        else:
            self.optimizer = optimizers.Adam(lr=self.lr, decay=self.decay)
        self.model = create_network(self.input_size, self.output_size,
                                    self.activation, self.dropout_rate, self.optimizer)

    def train(self, x_train, y_train, x_val=None, y_val=None, verbose=True):

        x_train = x_scaler.fit_transform(x_train)
        y_train = y_scaler.fit_transform(y_train)

        if x_val is not None:
            x_val = x_scaler.transform(x_val)
        if y_val is not None:
            y_val = y_scaler.transform(y_val)

        if (x_val is not None) and (y_val is not None):
            callbacks = [keras.callbacks.EarlyStopping(patience=10)]
            history = self.model.fit(x_train, y_train,
                                     validation_data=(x_val, y_val), epochs=self.epochs,
                                     callbacks=callbacks, batch_size=self.batch_size,
                                     verbose=verbose)
        else:
            history = self.model.fit(x_train, y_train,
                                     epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

        return history

    def predict(self, x):
        x = x_scaler.transform(x)
        y = self.model.predict(x)
        return y_scaler.inverse_transform(y)


def create_network(input_size, output_size, activation, dropout_rate, optimizer):
    model = Sequential([
        Dense(120, activation=activation, input_shape=(input_size,), name="dense_layer_1"),
        Dropout(dropout_rate, name="dropout_layer_1"),
        Dense(120, activation=activation, name="dense_layer_2"),
        Dropout(dropout_rate, name="dropout_layer_2"),
        Dense(120, activation=activation, name="dense_layer_3"),
        Dropout(dropout_rate, name="dropout_layer_3"),
        Dense(120, activation=activation, name="dense_layer_4"),
        Dropout(dropout_rate, name="dropout_layer_4"),
        Dense(output_size, activation="linear", name="output_layer"),
    ])

    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model
