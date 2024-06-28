# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Plot position heatmaps for training and test trajectories
"""

import sys

sys.path.insert(0, "../../")
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Project_Utils_ as Project_Utils

num_joints = 6

loading_folder = "./Experiments/"
training_num_sin = 100
test_num_sin = 100
training_range = 0.5
test_range = 1.4
filename_template = "SumOfSin__w_f0.02__num_sin{}__pos_range_{}__50seconds__seed{}.pkl"

# load dataframes
tr_data_pd_list = []
test_data_pd_list = []
for seed in range(11):
    # for seed in [6]:
    tr_filename = loading_folder + filename_template.format(training_num_sin, training_range, seed)
    # skip missing seeds
    if not os.path.isfile(tr_filename):
        continue
    test_filename = loading_folder + filename_template.format(test_num_sin, test_range, seed)
    tr_data_pd_list.append(pkl.load(open(tr_filename, "rb")))
    test_data_pd_list.append(pkl.load(open(test_filename, "rb")))

# concatenate dataframes
tr_data_pd = pd.concat(tr_data_pd_list)
test_data_pd = pd.concat(test_data_pd_list)

# extract positions
q_names = ["q_" + str(i) for i in range(1, num_joints + 1)]
dq_names = ["dq_" + str(i) for i in range(1, num_joints + 1)]
dq_names = ["ddq_" + str(i) for i in range(1, num_joints + 1)]

tr_q = tr_data_pd[q_names].to_numpy()
tr_dq = tr_data_pd[dq_names].to_numpy()
test_q = test_data_pd[q_names].to_numpy()
test_dq = test_data_pd[dq_names].to_numpy()

# quantize the position intervals
nbins = 20
q_min = np.floor(np.min(np.concatenate([tr_q, test_q]), axis=0) * 10) / 10
q_max = np.floor(np.max(np.concatenate([tr_q, test_q]), axis=0) * 10) / 10
q_intervals = np.array([np.linspace(q_min, q_max, nbins)]).squeeze()
dq_min = np.floor(np.min(np.concatenate([tr_dq, test_dq]), axis=0) * 10) / 10
dq_max = np.floor(np.max(np.concatenate([tr_dq, test_dq]), axis=0) * 10) / 10
dq_intervals = np.array([np.linspace(dq_min, dq_max, nbins)]).squeeze()


# compute the frequency
tr_q_frequency = np.zeros_like(q_intervals)
test_q_frequency = np.zeros_like(q_intervals)
lim_tr_list = []
lim_test_list = []
tr_dq_frequency = np.zeros_like(dq_intervals)
test_dq_frequency = np.zeros_like(dq_intervals)
lim_dq_tr_list = []
lim_dq_test_list = []

tr_q_dq_frequency = np.zeros([num_joints, nbins, nbins])
test_q_dq_frequency = np.zeros([num_joints, nbins, nbins])
lim_q_dq_tr_list = []
lim_q_dq_test_list = []

for ndof in range(num_joints):
    tr_q_frequency[:, ndof], lim_tr = np.histogram(
        tr_q[:, ndof], bins=nbins, range=(q_min[ndof], q_max[ndof]), density=True
    )
    test_q_frequency[:, ndof], lim_test = np.histogram(test_q[:, ndof], bins=nbins, range=(q_min[ndof], q_max[ndof]))
    lim_tr_list.append(lim_tr)
    lim_test_list.append(lim_test)

    tr_dq_frequency[:, ndof], lim_tr = np.histogram(tr_dq[:, ndof], bins=nbins, range=(dq_min[ndof], dq_max[ndof]))
    test_dq_frequency[:, ndof], lim_test = np.histogram(
        test_dq[:, ndof], bins=nbins, range=(dq_min[ndof], dq_max[ndof])
    )
    lim_dq_tr_list.append(lim_tr)
    lim_dq_test_list.append(lim_test)

    tr_q_dq_frequency[ndof, :, :], tr_xedges, tr_yedges = np.histogram2d(
        tr_dq[:, ndof], tr_dq[:, ndof], bins=nbins, range=[(dq_min[ndof], dq_max[ndof]), (dq_min[ndof], dq_max[ndof])]
    )
    test_q_dq_frequency[ndof, :, :], test_xedges, test_yedges = np.histogram2d(
        test_dq[:, ndof],
        test_dq[:, ndof],
        bins=nbins,
        range=[(dq_min[ndof], dq_max[ndof]), (dq_min[ndof], dq_max[ndof])],
    )
    lim_q_dq_tr_list.append((tr_xedges, tr_yedges))
    lim_q_dq_test_list.append((test_xedges, test_yedges))


# plot pos
fig = plt.figure(constrained_layout=True, figsize=(7, 4))
subfigsnest = fig.subfigures(2, 1, height_ratios=[1, 0.2])
subfigs = subfigsnest[0].subfigures(1, 6)
for dof, fig in zip(range(num_joints), subfigs):
    curr_fig = subfigs[dof]
    if dof % 2 == 0:
        curr_fig.set_facecolor("0.9")
    curr_fig.suptitle("joint " + str(dof + 1))
    axes = curr_fig.subplots(1, 2)
    map = axes[0].imshow(tr_q_frequency[:, dof].reshape(-1, 1), origin="lower", cmap="Reds")
    axes[0].set_xticks([])
    axes[0].set_ylim([-0.5, nbins - 0.5])
    axes[0].set_yticks((np.array(list(range(0, nbins + 1))) - 0.5)[::2])
    axes[0].set_yticklabels(np.round(lim_tr_list[0], 2)[::2], fontsize=9)
    # if dof == 0:
    #     axes[0].set_yticklabels(np.round(lim_tr_list[0],2)[::2], fontsize=9)
    # else:
    #     axes[0].set_yticklabels([])
    axes[0].set_xlabel("train")
    if dof == 0:
        axes[0].set_ylabel("$q$ intervals [rad]", fontsize=9)
    else:
        axes[0].set_ylabel(" ", fontsize=9)

    axes[1].imshow(test_q_frequency[:, dof].reshape(-1, 1), origin="lower", cmap="Reds")
    axes[1].set_xticks([])
    axes[1].set_ylim([-0.5, nbins - 0.5])
    axes[1].set_yticks((np.array(list(range(0, nbins + 1))) - 0.5)[::2])
    # axes[1].set_yticklabels(np.round(lim_tr_list[0],2)[::2], fontsize=9)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("test")
    # curr_fig.subplots_adjust(bottom=0.1, top=0.9, left=0.3, right=0.8,
    #                          wspace=0.1, hspace=0.5)

ylabel_ax = subfigsnest[0].add_axes([0.1, 0.1, 0.02, 0.8])

cb_ax = subfigsnest[1].add_axes([0.2, 0.5, 0.6, 0.2])
cbar = subfigsnest[1].colorbar(map, cax=cb_ax, orientation="horizontal")
cbar.set_ticks(np.arange(0, 0.75, 0.2))
cbar.set_ticklabels([str(int(x)) + "%" for x in np.arange(0, 0.75, 0.2) * 100])


# x = plt.colorbar(map, shrink=1, ax=axes, label='probability',  location='bottom')#, ticks=[0,500,1000, 1500])

# # plot pos x vel
# fig = plt.figure(constrained_layout=True, figsize=(7, 4.5))
# subfigs = fig.subfigures(2, 3).flatten()
# for dof, fig in zip(range(num_joints), subfigs):
#     curr_fig = subfigs[dof]
#     curr_fig.suptitle('joint ' + str(dof+1))
#     axes = curr_fig.subplots(1,2)
#     map = axes[0].imshow(tr_q_dq_frequency[dof, :, :], origin='lower', cmap='Blues')
#     # axes[0].set_xticks([])
#     # axes[0].set_ylim([-0.5,nbins-0.5])
#     # axes[0].set_yticks((np.array(list(range(0,nbins+1)))-0.5)[::2])
#     # axes[0].set_yticklabels(np.round(lim_tr_list[0],2)[::2], fontsize=9)
#     axes[0].set_xlabel('training')
#     axes[1].imshow(test_q_dq_frequency[dof, :, :], origin='lower', cmap='Blues')
#     # axes[1].set_xticks([])
#     # axes[1].set_ylim([-0.5,nbins-0.5])
#     # axes[1].set_yticks((np.array(list(range(0,nbins+1)))-0.5)[::2])
#     # # axes[1].set_yticklabels(np.round(lim_tr_list[0],2)[::2], fontsize=9)
#     # axes[1].set_yticklabels([])
#     axes[1].set_xlabel('test')
#     curr_fig.colorbar(map, ax=axes, shrink=0.6, location = 'bottom')

# # plot acc
# fig = plt.figure(constrained_layout=True, figsize=(7, 4.5))
# subfigs = fig.subfigures(2, 3).flatten()
# for dof, fig in zip(range(num_joints), subfigs):
#     curr_fig = subfigs[dof]
#     curr_fig.suptitle('joint ' + str(dof+1))
#     axes = curr_fig.subplots(1,2)
#     map = axes[0].imshow(tr_dq_frequency[:,dof].reshape(-1,1), origin='lower', cmap='Blues')
#     axes[0].set_xticks([])
#     axes[0].set_ylim([-0.5,nbins-0.5])
#     axes[0].set_yticks((np.array(list(range(0,nbins+1)))-0.5)[::2])
#     axes[0].set_yticklabels(np.round(lim_tr_list[0],2)[::2], fontsize=9)
#     axes[0].set_xlabel('training')
#     axes[0].set_ylabel('$\dot{q}$ range')
#     axes[1].imshow(test_dq_frequency[:,dof].reshape(-1,1), origin='lower', cmap='Blues')
#     axes[1].set_xticks([])
#     axes[1].set_ylim([-0.5,nbins-0.5])
#     axes[1].set_yticks((np.array(list(range(0,nbins+1)))-0.5)[::2])
#     # axes[1].set_yticklabels(np.round(lim_tr_list[0],2)[::2], fontsize=9)
#     axes[1].set_yticklabels([])
#     axes[1].set_xlabel('test')
#     curr_fig.colorbar(map, ax=axes, shrink=0.7, location = 'bottom', label='number of samples', ticks=[0,250, 500, 750, 1000])


plt.show()
