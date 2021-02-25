#!/usr/bin/python
# -*- coding: utf-8 -*-
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Author: Zhongsheng Chen
# Date: 05/09/2020
# Copyright: Copyright 2020, Beijing University of Chemical Technology
# License: The MIT License (MIT)
# Email: zschen@mail.buct.edu.cn
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os.path import basename
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as ticker
from scipy.stats import kde

from dataset import _magical_sinus


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def _get_scatter_data(inp, out):
    """
    A helper function that organize inputs and outputs applications.
    """
    xi, yi, zi = inp[:, 0], inp[:, 1], out
    return xi, yi, zi


def _plot_surface(ax, function):
    x_range = np.arange(0, 1, 0.01)
    y_range = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(x_range, y_range)
    Z = function(X, Y)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.winter(norm(Z))
    surf = ax.plot_surface(X, Y, Z,
                           rstride=5, cstride=5, facecolors=colors, shade=False)

    surf.set_facecolor((0, 0, 0, 0))
    return ax


def plot_training_curve(d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true, fig_dir="", save_fig=False):
    plt.plot(d_loss_err, label="Discriminator Loss")
    plt.plot(d_loss_true, label="Discriminator Loss - True")
    plt.plot(d_loss_fake, label="Discriminator Loss - Fake")
    plt.plot(g_loss_err, label="Generator Loss")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Loss")
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_cgan_loss.jpeg", bbox_inches='tight', dpi=300)
    plt.show()

    plt.plot(g_pred, label="Average Generator Prediction")
    plt.plot(g_true, label="Average Generator Reality")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Average Prediction")

    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_cgan_ave_pred.jpeg", dpi=300)
    plt.show()


def plot_dataset(X_train, X_test, y_train, y_test, exp_config, fig_dir):
    # plot 3-d surface for 3-d testing function.
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")
    ax = _plot_surface(ax, _magical_sinus)

    # plot 3-d scatter for applications points which are using for training and testing.
    xt, yt, zt = _get_scatter_data(X_train, y_train)
    ax.scatter3D(xt, yt, zt.flatten(), c='r', marker='o', s=10, label="training samples")
    xe, ye, ze = _get_scatter_data(X_test, y_test)
    ax.scatter3D(xe, ye, ze.flatten(), c='g', marker='s', s=10, label="validation samples")

    ax.set_zlim(0, 1.05 * max(zt.max(), ze.max()))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter(r'%.02f'))
    ax.set_xlabel(r"$x_1$"), ax.set_ylabel(r"$x_2$"), ax.set_zlabel(r"$f(x_1,y_2)$")
    ax.legend(loc="upper right")

    fig.tight_layout()
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_training_and_testing_data.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_ypred_with_locations(x, ytrue, ypred_regcgan, ypred_gp,
                              *, alpha=0.5, elevation=30, azimuth=60,
                              prefix="", fig_dir="", save_fig=False, legend=True, zlim=None, show=True):
    xdata = []
    ydata = []
    legend_str = []

    if ytrue is not None:
        xdata.append(x)
        ydata.append(ytrue)
        legend_str.append("True")

    if ypred_regcgan is not None:
        xdata.append(x)
        ydata.append(ypred_regcgan)
        legend_str.append("RegCGAN")

    if ypred_gp is not None:
        xdata.append(x)
        ydata.append(ypred_gp)
        legend_str.append("GP")
    n_subplots = len(xdata)
    fig = plt.figure(figsize=(18, 9))
    for i, (x, y, leg_str) in enumerate(zip(xdata, ydata, legend_str), 1):
        ax = fig.add_subplot(1, n_subplots, i, projection='3d')
        ax = _plot_surface(ax, _magical_sinus)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter(r'%.02f'))
        ax.set_xlabel(r"$x_1$"), ax.set_ylabel(r"$x_2$"), ax.set_zlabel(r"$f(x_1,y_2)$")

        xx, yy, zz = _get_scatter_data(x, y)
        if legend:
            ax.scatter3D(xx, yy, zz.flatten(), c='r', marker='o', alpha=alpha, label=leg_str)
            ax.legend(loc="lower right")
        else:
            ax.scatter3D(xx, yy, zz.flatten(), c='r', marker='o', alpha=alpha)

        if zlim is not None:
            ax.set_zlim(zlim)
        ax.view_init(elev=elevation, azim=azimuth)
    fig.tight_layout()
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_{prefix}_ypred_at_locations.jpeg", bbox_inches='tight', dpi=300)
    if show:
        plt.show()


def plot_density_cont(x, y, title="", n_bins=200, ylim_min=0, y_lim_max=0, prefix="0", fig_dir="", save_fig=False):
    x = x.flatten()
    y = y.flatten()
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():n_bins * 1j, y.min():y.max():n_bins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    if ylim_min or y_lim_max:
        plt.ylim(ylim_min, y_lim_max)
    plt.title(title)

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_{prefix}_contours.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_densities_joint(ytrue, ypred_regcgan, ypred_gp, title="", prefix="", fig_dir="",
                         save_fig=False, ylim=None):
    if ytrue is not None:
        sns.distplot(ytrue, hist=False, kde=True,
                     color='green', label="True",
                     kde_kws={'linestyle': 'solid'})

    if ypred_regcgan is not None:
        sns.distplot(ypred_regcgan, hist=False, kde=True,
                     color='violet', label="RegCGAN",
                     kde_kws={'linestyle': 'dashdot'})

    if ypred_gp is not None:
        sns.distplot(ypred_gp, hist=False, kde=True,
                     color='orange', label="GP",
                     kde_kws={'linestyle': 'dashed'})

    plt.xlabel(r"$y$")
    plt.ylabel("Probability density")
    plt.legend(loc="upper right")
    plt.title(title)
    if ylim is not None:
        plt.xlim(ylim)

    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_{prefix}.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_scatter_density(x, y, title):
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_den_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_den_y = fig.add_subplot(gs[1, 1], sharey=ax)

    def scatter_kde():
        ax.set_xlabel(r"$x_1$"), ax.set_ylabel(r"$x_2$")
        ax_den_x.tick_params(axis="x", labelbottom=False)
        ax_den_y.tick_params(axis="y", labelleft=False)

        ax.scatter(x, y, s=10, c="m", marker="o")
        sns.distplot(x, vertical=False, ax=ax_den_x)
        sns.distplot(y, vertical=True, ax=ax_den_y)

        ax.set_xlim([0, 1.25]), ax.set_ylim([0, 1.25])
        ax.set_aspect("equal", "box")

    scatter_kde()
    fig.suptitle(title)
    plt.show()


def plot_contour(give_y):
    x = np.arange(0, 1, 0.001)
    y = np.arange(0, 1, 0.001)
    X, Y = np.meshgrid(x, y)
    Z = _magical_sinus(X, Y)

    fig, ax = plt.subplots()
    counter = ax.contour(X, Y, Z, levels=[give_y], colors='b')
    ax.clabel(counter, colors='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal", "box")

    fig.tight_layout()
    plt.show()


def plot_sparse_regions(X, X_outliers, radius, fig_dir=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1],
               label="Original applications", edgecolors='g')
    ax.scatter(X_outliers[:, 0], X_outliers[:, 1],
               s=2000 * radius, edgecolors='r',
               facecolors='none', label='Outliers')
    lgnd = ax.legend(bbox_to_anchor=(0.5, -0.15), loc="center", ncol=2)

    for handle in lgnd.legendHandles:
        handle.set_sizes([40])

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    plt.savefig(f"{fig_dir}/{basename(fig_dir)}_sparse_regions.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_cvt_discrepancy(min_n_sampling, max_n_sampling, interval, X_full_discrepancy_list, fig_dir, title=""):
    x_tick_labels = [str(n_sampling) for n_sampling in range(min_n_sampling, max_n_sampling, interval)]

    x = np.arange(len(x_tick_labels))
    y = X_full_discrepancy_list
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, "g--o")
    ax.legend(loc="upper right")
    ax.set_xlabel("The number of CVT samples")
    ax.set_ylabel("Discrepancy")
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.savefig(f"{fig_dir}/{basename(fig_dir)}_discrepancy.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_voronoi_cvt(X_outliers, X_CVT, Voronoi, voronoi_plot_2d, fig_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X_outliers[:, 0], X_outliers[:, 1], 'ro', label="Outliers", markersize=5)
    ax.plot(X_CVT[:, 0], X_CVT[:, 1], 'gs', label="CVT samples", markersize=5)
    vor = Voronoi(X_CVT)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_alpha=0.1)
    ax.set_aspect("equal", "box")
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc="center", ncol=2)

    fig.tight_layout()
    plt.savefig(f"{fig_dir}/{basename(fig_dir)}_Voronoi_CVT_samples.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_ypred(ytrue, ypred_regcgan, ypred_gp, title="", prefix="", fig_dir="",
               save_fig=False, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if ytrue is not None:
        ax.plot(ytrue, label="True", color="green", linestyle="-", marker="o", markersize=5)
    if ypred_regcgan is not None:
        ax.plot(ypred_regcgan, label="RegCGAN", color="magenta", linestyle="-.", marker="v", markersize=5)
    if ypred_gp is not None:
        ax.plot(ypred_gp, label="GP", color="crimson", linestyle=":", marker="d", markersize=5)

    plt.xlabel(r"Index of samples")
    plt.ylabel(r"Outputs")
    plt.legend(loc="upper right")

    if title:
        plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_{prefix}_ypred.jpeg", bbox_inches='tight', dpi=300)
    plt.show()


def plot_mlp_ypred(ytrue, y_pred_baseline, y_pred_cvt,
                   y_pred_mtd, y_pred_ttd, y_pred_bootstrap,
                   y_pred_cgan, y_pred_psovsg,
                   title="", prefix="", fig_dir="",
                   save_fig=False, ylim=None, marker_size=5, line_width=1,
                   fig_width=20, fig_height=5):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if ytrue is not None:
        ax.plot(ytrue, label="True", color="green", linestyle="-",
                marker="o", markersize=marker_size, linewidth=line_width)
    if y_pred_baseline is not None:
        ax.plot(y_pred_baseline, label="No VS", color="magenta", linestyle="-.",
                marker="v", markersize=marker_size, linewidth=line_width)
    if y_pred_cvt is not None:
        ax.plot(y_pred_cvt, label="VS by our method ", color="blue", linestyle="-.",
                marker="s", markersize=marker_size, linewidth=line_width)
    if y_pred_mtd is not None:
        ax.plot(y_pred_mtd, label="VS by MTD", color="skyblue", linestyle="-.",
                marker="<", markersize=marker_size, linewidth=line_width)
    if y_pred_ttd is not None:
        ax.plot(y_pred_ttd, label="VS by TTD", color="olive", linestyle="--",
                marker=">", markersize=marker_size, linewidth=line_width)
    if y_pred_bootstrap is not None:
        ax.plot(y_pred_bootstrap, label="VS by Bootstrap", color="cyan", linestyle="--",
                marker="*", markersize=marker_size, linewidth=line_width)
    if y_pred_cgan is not None:
            ax.plot(y_pred_cgan, label="VS by CGAN", color="coral", linestyle=":",
                    marker="p", markersize=marker_size, linewidth=line_width)
    if y_pred_psovsg is not None:
            ax.plot(y_pred_psovsg, label="VS by PSO-VSG", color="teal", linestyle=":",
                    marker="d", markersize=marker_size, linewidth=line_width)

    plt.xlabel(r"Index of samples")
    plt.ylabel(r"Outputs")
    lg = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter(r'%.03f'))

    if title:
        plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}_{prefix}_mlp_ypred.jpeg", bbox_inches='tight',
                    bbox_extra_artists=(lg,), dpi=300)
    plt.show()
