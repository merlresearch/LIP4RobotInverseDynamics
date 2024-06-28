# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Config utilis
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
        Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
"""


boolean_variables = [
    "flg_load",
    "flg_save",
    "flg_train",
    "flg_norm",
    "flg_norm_noise",
    "flg_mean_norm",
    "single_sigma_n",
    "shuffle",
    "flg_frict",
    "flg_np",
    "flg_compute_acc",
    "flg_plot",
    "drop_last",
]

inv_variables = [
    "num_dof",
    "num_dat_tr",
    "downsampling",
    "downsampling_data_load",
    "batch_size",
    "n_epoch",
    "n_epoch_print",
    "num_threads",
    "num_dof",
    "num_prism",
    "num_rev",
    "num_par",
]

float_variables = ["vel_threshold", "sigma_n_num", "lr", "p_dyn_perc_error", "p_dyn_add_std"]


def get_d_param(config, kernel_name):
    d = {}
    for k, v in config.items(kernel_name):
        if k in boolean_variables:
            d[k] = config[kernel_name].getboolean(k)
        elif k in inv_variables:
            d[k] = config[kernel_name].getint(k)
        elif k in float_variables:
            d[k] = config[kernel_name].getfloat(k)
        else:
            d[k] = v
    return d
