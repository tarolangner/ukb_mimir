import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import sys

import scipy.stats
from sklearn.metrics import r2_score

def plotScatter(values_a, values_b, path_out, name, unit):

    f, ax = plt.subplots(1, 2, figsize=(8, 8/2))

    ax_cor = ax[0]
    ax_bland = ax[1]

    title = generateTitle(values_a, values_b)
    ax_cor.set_title(title[0])
    ax_bland.set_title(title[1])

    plotCorrelations(values_a, values_b, ax_cor, "C0", unit)
    plotBlandAltman(ax_bland, values_a, values_b, "C0", showLoa=True, unit=unit)

    ax_cor.set_ylabel(f"prediction in {unit}")
    ax_bland.set_ylabel(f"(reference - prediction) in {unit}")

    ax_cor.set_aspect('equal', 'box')
    ax_bland.set_aspect('auto', 'box')

    ax_bland.yaxis.tick_right()
    ax_bland.yaxis.set_label_position("right")

    plt.tight_layout()

    plt.savefig(path_out, dpi=200)

    plt.close()


def generateTitle(values_a, values_b):

    N = len(values_a)
    (mae, r, r2, loa_low, loa_high) = getAgreement(values_a, values_b)

    loa = "({0:0.2f} to {1:0.2f})".format(loa_low, loa_high)

    title = ["N: {0}, R$^2$: {1:0.3f}, MAE: {2:0.3f}".format(N, r2, mae), "r: {0:0.3f}, LoA: {1}".format(r, loa)]

    return title           


def getAgreement(gt, out):

    mask = np.invert(np.isnan(gt))
    gt = np.copy(gt[mask])
    out = np.copy(out[mask])

    mae = np.mean(np.abs(gt - out))
    (r, _) = scipy.stats.pearsonr(gt, out)
    r2 = r2_score(gt, out)

    dif = gt - out
    dif_mean = np.mean(dif)
    dif_std = np.std(dif, ddof=1)

    loa_low = dif_mean - 1.96 * dif_std
    loa_high = dif_mean + 1.96 * dif_std

    return (mae, r, r2, loa_low, loa_high)



def plotCorrelations(values_a, values_b, ax, c, unit):

    all_values = np.concatenate((values_b.flatten(), values_a.flatten()))
    value_max = max(all_values)
    value_min = min(all_values)
    value_range = value_max - value_min
    margin = 0.1 * value_range

    limits = np.array((value_min - margin, value_max + margin))

    plt.sca(ax)

    plt.xlabel(f"reference in {unit}")

    # Plot diagonal
    ax.plot([limits[0], value_max + margin], [limits[0], value_max + margin], c="k", linewidth=0.5)

    # Marker scales:
    s = scaleMarkers(0.25, values_a, values_b)

    ax.scatter(values_a, values_b, s=s, linewidths=0, c=c, zorder=20)

    ax.set_xlim(limits)
    ax.set_ylim(limits)


def scaleMarkers(base_s, values_a, values_b):

    dist = np.abs(values_a - values_b)
    dist = (dist - np.amin(dist)) / (np.amax(dist) - np.amin(dist))
    s = 3*(dist*base_s) + base_s

    return s


def plotBlandAltman(ax, values_a, values_b, c, showLoa, unit):

    plt.sca(ax)

    means = (values_a + values_b) / 2
    difs = values_a - values_b

    #
    mean = np.mean(difs)
    std = np.std(difs)

    # Marker scales:
    s = scaleMarkers(0.25, values_a, values_b)

    ax.scatter(means, difs, s=s, color=c, linewidths=0, zorder=20)

    x_lim = ax.get_xlim()

    std_mult = 1.96

    ax.plot(x_lim, np.array((mean, mean)), color="k", linestyle="--", linewidth=0.5)
    ax.plot(x_lim, np.array((mean + std_mult*std, mean + std_mult*std)), color="k", linestyle="-.", linewidth=0.5)
    ax.plot(x_lim, np.array((mean - std_mult*std, mean - std_mult*std)), color="k", linestyle="-.", linewidth=0.5)

    ax.plot(x_lim, np.array((0, 0)), color="#000000", alpha = 0.2) 

    #x_lim = (0, 40)
    ax.set_xlim(x_lim)

    plt.sca(ax)
    plt.xlabel(f"means in {unit}")

    if showLoa:
        # Text

        LoA_high = mean + std_mult * std
        LoA_low = mean - std_mult * std

        x_edge = x_lim[1] - (x_lim[1] - x_lim[0]) * 0.01

        y_lim = ax.get_ylim()
        y_shift = (y_lim[1] - y_lim[0]) * 0.01

        color="#000000"
        plt.text(x_edge, mean + std_mult*std + y_shift, "+{0} SD: {1:0.1f}".format(std_mult, LoA_high), color=color, ha="right", va="bottom")

        plt.text(x_edge, mean + y_shift, "mean: 0.0".format(mean), color=color, ha="right", va="bottom")

        plt.text(x_edge, mean - std_mult*std - y_shift, "-{0} SD: {1:0.1f}".format(std_mult, LoA_low), color=color, ha="right", va="top")
