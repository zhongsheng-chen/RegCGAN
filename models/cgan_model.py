#!/usr/bin/python
# -*- coding: utf-8 -*-
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Author: Zhongsheng Chen
# Date: 05/09/2020
# Copyright: Copyright 2020, Beijing University of Chemical Technology
# License: The MIT License (MIT)
# Email: zschen@mail.buct.edu.cn
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

"""" Conditional GAN for Regression.

We implement conditional GAN for regression, termed as RegCGAN.
It consists of a Generator G and a Discriminator D. Generator G will pass though x and z,
and will output y; however, Discriminator D will consume x and y and try to distinguish
either true y or fake one. The Generator G and the Discriminator D are trained simultaneously
to play a max-min games. Once training procedure finishes completely, we have that G(z, y)
approximate p(x, y). By fixing y, we have G(z|y) approximating p(x|y).
By sampling z, we can therefore obtain samples following approximately p(x|y),
which is the predictive distribution of x for a new observation y.

"""

from __future__ import print_function, division

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from network import build_discriminator, build_generator


class RegCGAN():
    """ RegCGAN class

    The architecture for G and D is separately defined in network.py.
    The G and D are optimized by stochastic gradient optimizers like Adam and SGD. Our codes
    derive from contributions at https://github.com/eriklindernoren/Keras-GAN and
    https://github.com/mkirchmeyer/ganRegression

    """

    def __init__(self, exp_config):
        if exp_config.model.optim_gen == "Adam":
            self.optimizer_gen = Adam(exp_config.model.lr_gen, decay=exp_config.model.dec_gen)
        else:
            self.optimizer_gen = SGD(exp_config.model.lr_gen, decay=exp_config.model.dec_gen)
        if exp_config.model.optim_disc == "Adam":
            self.optimizer_disc = Adam(exp_config.model.lr_disc, decay=exp_config.model.dec_disc)
        else:
            self.optimizer_disc = SGD(exp_config.model.lr_disc, decay=exp_config.model.dec_disc)
        self.activation = exp_config.model.activation
        self.seed = exp_config.model.random_seed
        self.scenario = exp_config.dataset.scenario

        if self.scenario == "magical_sinus":
            self.x_input_size = 2
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 1
        elif self.scenario == "hdpe":
            self.x_input_size = 15
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 2
        else:
            self.x_input_size = 2
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 1

        if exp_config.model.architecture is not None:
            self.architecture = exp_config.model.architecture

        self.discriminator = build_discriminator(self)  # Build  the Discriminator D
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer_disc,
            metrics=['accuracy'])

        self.generator = build_generator(self)  # Build the Generator G
        noise = Input(shape=(self.z_input_size,))
        x = Input(shape=(self.x_input_size,))
        y = self.generator([noise, x])

        self.discriminator.trainable = False  # For the combined model, we will only train the generator
        validity = self.discriminator([x, y])
        self.combined = Model([noise, x], validity)  # The combined model will train generator to fool discriminator
        self.combined.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer_gen)

        # Print network's architecture
        print(self.generator.summary())
        print(self.discriminator.summary())

    def train(self, xtrain, ytrain, epochs, batch_size=128, verbose=True):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        dLossErr = np.zeros([epochs, 1])
        dLossReal = np.zeros([epochs, 1])
        dLossFake = np.zeros([epochs, 1])
        gLossErr = np.zeros([epochs, 1])
        genPred = np.zeros([epochs, 1])
        genReal = np.zeros([epochs, 1])

        for epoch in range(epochs):

            # train Generator G and Discriminator D alternatively.
            for iter in range(int(xtrain.shape[0] // batch_size)):
                # -----------------------
                #  Train Discriminator D
                # -----------------------

                idx = np.random.randint(0, xtrain.shape[0], batch_size)  # Select a random half batch of samples
                true_x, true_y = xtrain[idx], ytrain[idx]

                noise = np.random.normal(0, 1, (batch_size, self.z_input_size))  # sample noise z from N(0, I)
                # Generate new observations of y
                fake_y = self.generator.predict([noise, true_x])
                # Train the discriminator on batch
                d_loss_real = self.discriminator.train_on_batch([true_x, true_y], valid)
                d_loss_fake = self.discriminator.train_on_batch([true_x, fake_y], fake)
                d_loss = np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator G
                # ---------------------

                idx = np.random.randint(0, xtrain.shape[0], batch_size)  # Condition on x
                true_x = xtrain[idx]
                g_loss = self.combined.train_on_batch([noise, true_x], valid)

            dLossErr[epoch] = d_loss[0]
            dLossReal[epoch] = d_loss_real[0]
            dLossFake[epoch] = d_loss_fake[0]
            gLossErr[epoch] = g_loss

            if verbose:
                print(f"Epoch: {epoch} / dLoss: {d_loss[0]} / gLoss: {g_loss}")

            ypred = self.predict(xtrain)
            genPred[epoch] = np.average(ypred)
            genReal[epoch] = np.average(ytrain)

        return dLossErr, dLossReal, dLossFake, gLossErr, genPred, genReal

    def predict(self, x_test):
        noise = np.random.normal(0, 1, (x_test.shape[0], self.z_input_size))
        y_pred = self.generator.predict([noise, x_test])
        return y_pred

    def sampling(self, x, n_sampling):
        y = self.predict(x)
        for i in range(n_sampling - 1):
            y_pred = self.predict(x)
            y = np.hstack([y, y_pred])
        mean = []
        for j in range(y.shape[0]):
            mean.append(np.mean(y[j, :]))

        return np.array(mean).reshape(-1, 1)
