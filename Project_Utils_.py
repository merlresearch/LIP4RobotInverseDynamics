# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
File with auxiliary functions

Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_estimate(Y, Y_hat, joint_index_list, Y_noiseless=None, Y_hat_var=None, output_name="", Y_prior_mean=None):
    """Prints estimates"""
    num_samples, num_joint = Y.shape
    for i in range(0, num_joint):
        plt.figure()
        plt.plot(Y[:, i], "r", label=output_name + "_" + str(joint_index_list[i]))
        plt.plot(Y_hat[:, i], "b", label=output_name + "_" + str(joint_index_list[i]) + "_hat")
        if Y_hat_var is not None:
            X_indices = np.arange(0, num_samples)
            std = 3 * np.sqrt(Y_hat_var[:, i])
            plt.fill(
                np.concatenate([X_indices, np.flip(X_indices)]),
                np.concatenate([Y_hat[:, i] + std, np.flip(Y_hat[:, i] - std)]),
                "b",
                alpha=0.4,
            )
        if Y_noiseless is not None:
            plt.plot(Y_noiseless[:, i], "k", label=output_name + "_" + str(joint_index_list[i]) + "_noiseless")
        if Y_prior_mean is not None:
            plt.plot(Y_prior_mean[:, i], "g", label=output_name + "_" + str(joint_index_list[i]) + "_prior_mean")
        plt.grid()
        plt.legend()


def print_estimate_single_fig(
    Y, Y_hat, joint_index_list, Y_noiseless=None, Y_hat_var=None, output_name="", Y_prior_mean=None, title=""
):
    """Prints estimates in a single figure with multiple subplots"""
    num_samples, num_joint = Y.shape
    num_rows_plot = int(np.ceil(num_joint / 2))

    fig = plt.figure(figsize=(9, 6))
    plt.suptitle(title)
    for i in range(0, num_joint):
        plt.subplot(num_rows_plot, 2, i + 1)
        plt.plot(Y[:, i], "r", label="True")  # =output_name+'_'+str(joint_index_list[i]))
        plt.plot(Y_hat[:, i], "b", label="Estimated")  # output_name+'_'+str(joint_index_list[i])+'_hat')
        plt.ylabel(r"$\tau_" + str(joint_index_list[i]) + "$")
        if Y_hat_var is not None:
            X_indices = np.arange(0, num_samples)
            std = 3 * np.sqrt(Y_hat_var[:, i])
            plt.fill(
                np.concatenate([X_indices, np.flip(X_indices)]),
                np.concatenate([Y_hat[:, i] + std, np.flip(Y_hat[:, i] - std)]),
                "b",
                alpha=0.4,
            )
        if Y_noiseless is not None:
            plt.plot(Y_noiseless[:, i], "k", label=output_name + "_" + str(joint_index_list[i]) + "_noiseless")
        if Y_prior_mean is not None:
            plt.plot(Y_prior_mean[:, i], "g", label=output_name + "_" + str(joint_index_list[i]) + "_prior_mean")
        plt.grid()
        plt.legend()
    fig.tight_layout()


def print_estimate_with_vel(
    Y, Y_hat, joint_index_list, dq, Y_noiseless=None, Y_hat_var=None, output_name="", Y_prior_mean=None
):
    """Prints estimates"""
    num_samples, num_joint = Y.shape
    for i in range(0, num_joint):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(Y[:, i], "r", label=output_name + "_" + str(joint_index_list[i]))
        plt.plot(Y_hat[:, i], "b", label=output_name + "_" + str(joint_index_list[i]) + "_hat")
        if Y_hat_var is not None:
            X_indices = np.arange(0, num_samples)
            std = 3 * np.sqrt(np.abs(Y_hat_var[:, i]))
            plt.fill(
                np.concatenate([X_indices, np.flip(X_indices)]),
                np.concatenate([Y_hat[:, i] + std, np.flip(Y_hat[:, i] - std)]),
                "b",
                alpha=0.4,
            )
        if Y_noiseless is not None:
            plt.plot(Y_noiseless[:, i], "k", label=output_name + "_" + str(joint_index_list[i]) + "_noiseless")
        if Y_prior_mean is not None:
            plt.plot(Y_prior_mean[:, i], "g", label=output_name + "_" + str(joint_index_list[i]) + "_prior_mean")
        plt.grid()
        plt.legend()
        # plot vel
        plt.subplot(2, 1, 2)
        plt.plot(dq[:, i], label="dq_" + str(joint_index_list[i]))
        plt.grid()
        plt.legend()


def get_stat_estimate(Y, Y_hat, stat_name="MSE", flg_print=True):
    """Prints the estimate performances"""
    # scompute the stat
    if stat_name == "MSE":
        stat = np.mean((Y - Y_hat) ** 2, 0)
    if stat_name == "nMSE":
        stat = np.mean((Y - Y_hat) ** 2, 0) / np.var(Y, 0)
    # print stats
    if flg_print:
        print(stat_name + ":", stat)
    return stat


def get_phi_labels(joint_index_list, num_dof, num_par_dyn):
    """Returns a list with the phi labels, given the list of joint indices, num_dof and num_par_dyn"""
    return [
        "phi_" + str(joint_index) + "_" + str(j + 1)
        for j in range(0, num_par_dyn * num_dof)
        for joint_index in joint_index_list
    ]


def get_data_from_features(
    data_frame_pkl, input_features, input_features_joint_list, output_feature, num_dof, joint_name_list=None
):
    """Function that returns:
    - a np matrix whose columns are the "input_features" elements of the pandas data pointed by data_frame_pkl
    - a np matrix containing the output vectors (the i-th conlumn corresponds to the output of the i-th link)
    - a list containing the active_dims list of each link gp
    """
    # get input output data
    if joint_name_list is None:
        joint_name_list = [str(i) for i in range(1, num_dof + 1)]
    data_frame = pd.read_pickle(data_frame_pkl)
    input_vector = data_frame[input_features].values
    output_labels = [output_feature + "_" + joint_name for joint_name in joint_name_list]
    output_vector = data_frame[output_labels].values
    # get the active dims of each joint
    active_dims_list = []
    for joint_index in range(0, len(joint_name_list)):
        active_dims_list.append(
            np.array([input_features.index(name) for name in input_features_joint_list[joint_index]])
        )
    return input_vector, output_vector, active_dims_list, data_frame


def get_inv_dyn_data_with_angle_transformation(
    data_frame_pkl,
    q_names,
    dq_names,
    ddq_names,
    tau_names,
    tau_matlab_names=None,
    prism_indices=None,
    revolute_indices=None,
):
    """
    Function that returns the IO of the inv_dyn with sin and cos of angles instead of angles, to consider periodicity:
    """
    # init the data list
    data_list = []
    num_dof = len(q_names)
    # get the dataframe
    data_frame = pd.read_pickle(data_frame_pkl)
    # add position data
    if prism_indices is not None:
        data_list.append(data_frame[q_names].values[:, prism_indices])
    if revolute_indices is not None:
        q_rev = data_frame[q_names].values[:, revolute_indices]
        data_list.append(np.cos(q_rev))
        data_list.append(np.sin(q_rev))
    # add velocity and acc data
    data_list.append(data_frame[dq_names].values)
    data_list.append(data_frame[ddq_names].values)
    # get the input
    input_data = np.concatenate(data_list, 1)
    # get the output
    output = data_frame[tau_names].values
    if tau_matlab_names is not None:
        output = data_frame[tau_names].values - data_frame[tau_matlab_names].values
    # get active dims list
    active_dims_list = [np.arange(0, input_data.shape[1])] * num_dof
    return input_data, output, active_dims_list, data_frame


def get_forward_dyn_data_with_angle_transformation(
    data_frame_pkl, q_names, dq_names, ddq_names, tau_names, prism_indices=None, revolute_indices=None
):
    """
    Function that returns the IO of the forward_dyn with sin and cos of angles instead of angles,
    to consider periodicity:
    """
    # init the data list
    data_list = []
    num_dof = len(q_names)
    # get the dataframe
    data_frame = pd.read_pickle(data_frame_pkl)
    # add position data
    if prism_indices is not None:
        data_list.append(data_frame[q_names].values[:, prism_indices])
    if revolute_indices is not None:
        q_rev = data_frame[q_names].values[:, revolute_indices]
        data_list.append(np.cos(q_rev))
        data_list.append(np.sin(q_rev))
    # add velocity and generalized torques
    data_list.append(data_frame[dq_names].values)
    data_list.append(data_frame[tau_names].values)
    # get the input
    input_data = np.concatenate(data_list, 1)
    # get the output
    output = data_frame[ddq_names].values
    # get active dims list
    active_dims_list = [np.arange(0, input_data.shape[1])] * num_dof
    return input_data, output, active_dims_list, data_frame


def get_dataset_poly_from_structure(data_frame_pkl, num_dof, output_feature, robot_structure, features_name_list=None):
    """Returns dataset and kernel info from robot structure:
    Robot structure is a list of num_dof elements containing:
    - 0 when the joint is revolute
    - 1 when the joint is prismatic
    As regards the positions cos(q), sin(q) are considered when the joint ir revolute (q when prism)
    """
    # load the pandas dataframe
    data_frame = pd.read_pickle(data_frame_pkl)
    # list with data
    data_list = []
    # lists with the active dims
    active_dims_acc_vel = []
    active_dims_friction = []
    active_dims_acc = []
    active_dims_q_dot = []
    active_dims_vel = []
    active_dims_mon_rev = []
    active_dims_mon_rev_cos = []
    active_dims_mon_rev_sin = []
    active_dims_mon_prism = []
    # init counters
    index_active_dim = 0
    num_rev = 0
    num_prism = 0
    # set names_list
    if features_name_list is None:
        features_name_list = [str(i) for i in range(1, num_dof + 1)]
    # get pos features
    for joint_index in range(0, num_dof):
        # get the q label
        q_label = "q_" + features_name_list[joint_index]
        # check the type of joint
        if robot_structure[joint_index] == 0:
            # when the type is revolute add cos(q) and sin(q)
            data_list.append(np.cos(data_frame[q_label].values).reshape([-1, 1]))
            active_dims_mon_rev_cos.append(np.array([index_active_dim]))
            index_active_dim += 1
            data_list.append(np.sin(data_frame[q_label].values).reshape([-1, 1]))
            active_dims_mon_rev_sin.append(np.array([index_active_dim]))
            active_dims_mon_rev.append(np.array([index_active_dim - 1, index_active_dim]))
            index_active_dim += 1
            num_rev += 2
        else:
            # when the type is prismatic add q
            data_list.append(data_frame[q_label].values.reshape([-1, 1]))
            active_dims_mon_prism.append(np.array([index_active_dim]))
            index_active_dim += 1
            num_prism += 1
    # get acc/vel/frictions features
    for joint_index_1 in range(0, num_dof):
        # add acc
        data_list.append(data_frame["ddq_" + features_name_list[joint_index_1]].values.reshape([-1, 1]))
        active_dims_acc_vel.append(index_active_dim)
        active_dims_acc.append(index_active_dim)
        index_active_dim += 1
        # add q_dot/friction features
        vel_label_1 = "dq_" + features_name_list[joint_index_1]
        data_list.append((data_frame[vel_label_1].values).reshape([-1, 1]))
        data_list.append(np.sign(data_frame[vel_label_1].values).reshape([-1, 1]))
        active_dims_friction.append(np.array([index_active_dim, index_active_dim + 1]))
        active_dims_q_dot.append(index_active_dim)
        index_active_dim += 2
        # add vel features
        for joint_index_2 in range(joint_index_1, num_dof):
            vel_label_2 = "dq_" + features_name_list[joint_index_2]
            data_list.append((data_frame[vel_label_1].values * data_frame[vel_label_2].values).reshape([-1, 1]))
            active_dims_acc_vel.append(index_active_dim)
            active_dims_vel.append(index_active_dim)
            index_active_dim += 1
    # get input output
    X = np.concatenate(data_list, 1)
    Y = data_frame[[output_feature + "_" + features_name_list[joint_index] for joint_index in range(0, num_dof)]].values
    # build the active dims diictionary
    active_dims_dict = dict()
    active_dims_dict["active_dims_mon_rev"] = active_dims_mon_rev
    active_dims_dict["active_dims_mon_rev_cos"] = active_dims_mon_rev_cos
    active_dims_dict["active_dims_mon_rev_sin"] = active_dims_mon_rev_sin
    active_dims_dict["active_dims_mon_prism"] = active_dims_mon_prism
    active_dims_dict["active_dims_acc_vel"] = np.array(active_dims_acc_vel)
    active_dims_dict["active_dims_acc"] = np.array(active_dims_acc)
    active_dims_dict["active_dims_q_dot"] = np.array(active_dims_q_dot)
    active_dims_dict["active_dims_vel"] = np.array(active_dims_vel)
    active_dims_dict["active_dims_friction"] = active_dims_friction
    return X, Y, active_dims_dict, data_frame


def normalize_signals(signals, norm_coeff=None, flg_mean=False, mean_coef=0.0):
    """Normalize signals: constraint the module of the signal
    between zero and one"""
    if norm_coeff is None:
        if flg_mean:
            # remove the mean
            mean_coef = np.mean(signals, 0)
            norm_coeff = (np.abs(signals - mean_coef)).max(axis=0)
            norm_coeff[norm_coeff == 0] = 1.0
        else:
            norm_coeff = (np.abs(signals)).max(axis=0)
            norm_coeff[norm_coeff == 0] = 1.0
            mean_coef = np.zeros(norm_coeff.size)
    return (signals - mean_coef) / norm_coeff, norm_coeff, mean_coef


def normalize_signals_(signals, norm_coeff=None):
    """Normalize signals: divide by the maximum  value of the signals"""
    if norm_coeff is None:
        norm_coeff = np.ones(signals.shape[1]) * ((np.abs(signals)).max())
    return signals / norm_coeff, norm_coeff


def denormalize_signals(signals, norm_coeff, mean_coef=0.0):
    """Denormalize signals"""
    return signals * norm_coeff + mean_coef


def get_results_dict(Y, Y_hat, norm_coef, mean_coef=0.0, Y_var=None, Y_noiseless=None, m_X=None, dq=None):
    """
    Create a dictionaty with denormalized data
    """
    d = {}
    norm_coef_array = np.array(norm_coef)
    # denormalize and save data
    d["Y"] = Y * norm_coef_array + mean_coef
    d["Y_hat"] = Y_hat * norm_coef_array + mean_coef
    if Y_var is not None:
        d["Y_var"] = Y_var * norm_coef_array**2
    if Y_noiseless is not None:
        d["Y_noiseless"] = Y_noiseless * norm_coef_array + mean_coef
    if m_X is not None:
        d["m_X"] = m_X * norm_coef_array + mean_coef
    if dq is not None:
        d["dq"] = dq
    return d


def plot_list(data_list, labels, titles):
    n = len(data_list)
    plt.figure()
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(data_list[i], label=labels[i])
        plt.title(titles[i])
        plt.grid()
        plt.legend()


def plot_eigenvals(eig_list, labels, colors, title="Eignevalues"):
    plt.figure()
    for i in range(len(eig_list)):
        for j in range(eig_list[i].shape[1]):
            if j == 0:
                plt.plot(np.sort(eig_list[i])[:, j], label=labels[i], color=colors[i])
            else:
                plt.plot(np.sort(eig_list[i])[:, j], color=colors[i])
    plt.title(title)
    plt.grid()
    plt.legend()


def plot_energy(true_kin, true_pot, true_lag, est_kin, est_pot, est_lag, title="Energy"):
    plt.figure(figsize=(9, 6))
    plt.subplot(3, 1, 1)
    plt.ylabel("Kinetic")
    if true_kin is not None:
        plt.plot(true_kin, label="True", c="r")
    plt.plot(est_kin, label="Estimated", c="b")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.ylabel("Potential")
    if true_pot is not None:
        plt.plot(true_pot, label="True", c="r")
    plt.plot(est_pot, label="Estimated", c="b")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.ylabel("Lagrangian")
    if true_lag is not None:
        plt.plot(true_lag, label="True", c="r")
    plt.plot(est_lag, label="Estimated", c="b")
    plt.grid()
    plt.legend()
    plt.suptitle(title)


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
