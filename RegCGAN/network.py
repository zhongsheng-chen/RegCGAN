#!/usr/bin/python
# -*- coding: utf-8 -*-
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Author: Zhongsheng Chen
# Date: 05/09/2020
# Copyright: Copyright 2020, Beijing University of Chemical Technology
# License: The MIT License (MIT)
# Email: zschen@mail.buct.edu.cn
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import keras
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, concatenate


def build_generator(network):
    seed = network.seed
    random_normal = keras.initializers.RandomNormal(seed=seed)

    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU(0.2)
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    if network.architecture == 1:  # Generator's architecture for magical_sinus
        x = Input(shape=(network.x_input_size,), dtype='float', name="Generator_input_x")
        x_output = Dense(20, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,), name="Generator_input_z")
        noise_output = Dense(20, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])
        output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output, name="Generator")
    elif network.architecture == 2:  # Generator's architecture for hdpeuce
        x = Input(shape=(network.x_input_size,), dtype='float', name="Generator_input_x")
        x_output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,), name="Generator_input_z")
        noise_output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output, name="Generator")
    else:
        raise NotImplementedError("Architecture does not exist")

    return model


def build_discriminator(network):
    seed = network.seed
    random_uniform = keras.initializers.RandomUniform(seed=seed)

    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    if network.architecture == 1:  # Discriminator's architecture for magical_sinus
        x = Input(shape=(network.x_input_size,), dtype='float', name="Discriminator_input_x")
        x_output = Dense(20, activation=activation, kernel_initializer=kerner_initializer)(x)

        y = Input(shape=(network.y_input_size,), name="Discriminator_input_y")
        y_output = Dense(20, activation=activation, kernel_initializer=kerner_initializer)(y)

        concat = concatenate([x_output, y_output])
        output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(output)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(output)

        model = Model(inputs=[x, y], outputs=validity, name="Discriminator")
    elif network.architecture == 2:  # Discriminator's architecture for hdpeuce
        x = Input(shape=(network.x_input_size,), dtype='float', name="Discriminator_input_x")
        x_output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(x)

        y = Input(shape=(network.y_input_size,), name="Discriminator_input_y")
        y_output = Dense(60, activation=activation, kernel_initializer=kerner_initializer)(y)

        concat = concatenate([x_output, y_output])
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(120, activation=activation, kernel_initializer=kerner_initializer)(output)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(output)

        model = Model(inputs=[x, y], outputs=validity, name="Discriminator")

    else:
        raise NotImplementedError("Architecture does not exist")

    return model
