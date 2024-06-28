# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Plot nMSE
Author: Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
"""
import sys

sys.path.insert(0, "../../")

import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import Project_Utils_ as Project_Utils


def plot_nMSE_boxplot(nMSE_list):
    ndof = 7
    ncols_fig = int(np.ceil(ndof / 2))
    num_tests = len(nMSE_list)
    num_models = len(nMSE_list[0])
    num_seeds = len(nMSE_list[0][0])

    for test in range(num_tests):
        fig, axes = Project_Utils.generate_subplots(
            ndof, sharex=False, sharey="row", row_wise=True, figsize=(7, 7), horizontal=False
        )
        # fig, axes = Project_Utils.generate_subplots(ndof,sharex=False, sharey=False,row_wise=True, figsize=(7, 7), horizontal=False)
        for dof, ax in zip(range(ndof), axes):
            ax.set_title("joint " + str(dof + 1), fontsize=10)
            ax.set_yscale("log")
            ax.minorticks_off()
            ax.set_xlim(0.65, 3.35)
            ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
            ax.set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
            # # ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
            # # ax.set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
            # if dof == 0 or dof == ncols_fig:
            if dof % 2 == 0:
                ax.set_ylabel("nMSE %")
            ax.grid("minor")
            # data = [np.array(nMSE_list[test][i]) for i in range(num_models)]
            data = [np.array([nmse[dof] for nmse in nMSE_list[test][i]]) for i in range(num_models)]
            labels = x_names[:num_models]
            x = 1 + 0.5 * (np.arange(num_models))
            ax.set_xlim(0.65, x[-1] + 0.35)
            bplots = ax.boxplot(
                data,
                positions=x,
                widths=0.3,
                patch_artist=True,
                labels=labels,
                showfliers=False,
                medianprops={"color": "red", "linewidth": 1.0},
                whiskerprops={"color": "black", "linewidth": 1.0},
                capprops={"color": "black", "linewidth": 1.0},
            )
            for patch, color in zip(bplots["boxes"], colors):
                patch.set_facecolor(color)
        # x = [1,2]
        # labels = ['GIP', 'RBF']
        ax.set_xticks(x, labels)
        fig.tight_layout()
        plt.savefig("PANDA_" + test_names[test] + ".pdf")


# matplotlib parameters
font = {"family": "normal", "size": 9}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
fontsize = 9

# test parameters
model_names = [
    "model_m_ind_RBF",
    "model_m_ind_GIP_with_friction",
    # 'with_NP_model_LK_POLY_RBF_sum',
    # 'with_NP_model_LK_RBF_1_sc',
    "with_linear_friction_model_LK_RBF_1_sc",
    "with_NP_model_LK_GIP_sum",
]
results_folder = "Results/"
test_file_prefix = "panda7dof_num_sin_100_seed_"  # {}_as_filtered_fcut4.0'
file_prefix_list = ["50-100-sin_two-dataset_"]
test_names = ["50-100"]

comb_list = list(range(12))
test_seed_list = [21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

# plot parameters
x_names = ["SE", "GIP", "LSE", "LIP", "Del"]
# colors = ['#cc0000', '#ff9900', '#cccc00', '#006600']
colors = ["red", "yellow", "green", "blue", "purple"]

# load torque files and plot
nMSE_list = []
for test_index, test_name in enumerate(test_names):
    print("\nTEST NAME:", test_name)
    # get results
    nMSE_models = []
    for model_index, model_name in enumerate(model_names):
        print("Model name:", model_name)
        nMSE_seeds = []
        # load data
        for comb_index in comb_list:
            for test_seed in test_seed_list:
                test_file_name = test_file_prefix + str(test_seed)
                file_name = (
                    results_folder
                    + file_prefix_list[test_index]
                    + "comb"
                    + str(comb_index)
                    + "_"
                    + model_name
                    + "_"
                    + test_file_name
                    + "_as_filtered_fcut4.0_estimates.pkl"
                )
                d = pkl.load(open(file_name, "rb"))
                # compute residuals
                E = d["Y"] - d["Y_hat"]
                nMSE_seeds.append(np.mean(E**2, 0) / np.var(d["Y"], 0))
                # print('nMSE_seeds[-1]', nMSE_seeds[-1])
        print("nMSE mean:", 100 * np.mean(np.concatenate([nmse.reshape([1, -1]) for nmse in nMSE_seeds], 0), 0))
        nMSE_models.append(nMSE_seeds)
    nMSE_list.append(nMSE_models)
    # plot

plot_nMSE_boxplot(nMSE_list)
plt.show()
