#!/usr/bin/python
# -*- coding: utf-8 -*-
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Author: Zhongsheng Chen
# Date: 05/09/2020
# Copyright: Copyright 2020, Beijing University of Chemical Technology
# License: The MIT License (MIT)
# Email: zschen@mail.buct.edu.cn
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import numpy as np
from sklearn.model_selection import train_test_split


def gen_data_magical_sinus(n_instance):
    """
    Generate n_instance of samples from a modified sinus function, noted by
    mdf_sinus here.
    """

    def _randrange(n, vmin, vmax):
        """
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        """
        return (vmax - vmin) * np.random.rand(n) + vmin

    x1 = _randrange(n_instance, 0, 1)
    x2 = _randrange(n_instance, 0, 1)

    scale = 0
    noise = np.random.normal(0, 0.1, n_instance)

    X = np.column_stack((x1, x2))
    y = _magical_sinus(x1, x2) + scale * noise
    y = y.reshape((n_instance, 1))
    return X, y


def _magical_sinus(x1, x2):
    """
    Create a noise-contaminated single-valued benchmarking function:
                 z = f(x, y)
    derived from sinus function. It feeds two variables and
    returns a single value for each given pair of inputs(x, y).
    """
    y = (1.3356 * (1.5 * (1 - x1))
         + (np.exp(2 * x1 - 1) * np.sin(3 * np.pi * (x1 - 0.6) ** 2))
         + (np.exp(3 * (x2 - 0.5)) * np.sin(4 * np.pi * (x2 - 0.9) ** 2)))
    return y


def get_dataset(n_instance=1000, scenario="magical_sinus", seed=1):
    """
    Create regression applications
    """

    if scenario == "magical_sinus":

        X_train, y_train = gen_data_magical_sinus(n_instance)
        X_val, y_val = gen_data_magical_sinus(n_instance)

    elif scenario == "hdpe":
        my_data_train = np.genfromtxt(f"../applications/hdpe/hdpe.applications", delimiter=',')
        X = my_data_train[:, :-1]
        y = my_data_train[:, -1]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)
    else:
        raise NotImplementedError("Dataset does not exist")

    return X_train, y_train.reshape(-1, 1), X_val, y_val.reshape(-1, 1)


def get_true_x_give_y(given_y=0.5, tolerance=0.01, num_realizations=10000000, *, X=None, y=None):
    if (X is None) or (y is None):
        X, y = gen_data_magical_sinus(num_realizations)

    data_points = np.concatenate((X, y), axis=1)
    true_x_give_y = data_points[((y.squeeze() > given_y - tolerance / 2) *
                                 (y.squeeze() < given_y + tolerance / 2)), :-1]
    return true_x_give_y


def get_true_y_given_x(given_x1=0.5, given_x2=0.5, tolerance=0.05, num_realizations=10000000, *, X=None, y=None):
    if (X is None) or (y is None):
        X, y = gen_data_magical_sinus(num_realizations)

    x1, x2 = X[:, 0], X[:, 1]
    cond_x1 = np.logical_and(x1 < given_x1 + tolerance, x1 > given_x1 - tolerance)
    cond_x2 = np.logical_and(x2 < given_x2 + tolerance, x2 > given_x2 - tolerance)

    true_y_given_x = y[cond_x1 * cond_x2]
    return true_y_given_x
