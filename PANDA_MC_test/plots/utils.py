# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import math

import matplotlib.pyplot as plt
import numpy as np


def choose_subplot_dimensions(k, horizontal=False):
    # https://stackoverflow.com/questions/28738836/matplotlib-with-odd-number-of-subplots
    if k < 4:
        a = k
        b = 1
        # return k, 1
    elif k < 11:
        a = math.ceil(k / 2)
        b = 2
        # return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        a = math.ceil(k / 3)
        b = 3

    if horizontal:
        return b, a
    return a, b


def generate_subplots(k, row_wise=True, sharex=True, sharey=False, ncol=None, nrow=None, horizontal=False, **fig_kw):
    # https://stackoverflow.com/questions/28738836/matplotlib-with-odd-number-of-subplots
    if nrow is None or ncol is None:
        nrow, ncol = choose_subplot_dimensions(k, horizontal=horizontal)
    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey, **fig_kw)

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=("C" if row_wise else "F"))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        return figure, axes
