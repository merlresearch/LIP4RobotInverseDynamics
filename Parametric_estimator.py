# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Inverse dynamics parametric estimator based on basis function reduction

Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""

import argparse
import configparser
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_files.Utils as Config_Utils
import Project_Utils_ as Project_Utils
import RealData.MELFA_RV4FRL.Hb as Hb

# UTILS FUNCTION #


def get_PHI(q_vect, dq_vect, ddq_vect, num_par):
    """Computes the PHI regression matrix"""
    num_samples, num_dof = q_vect.shape
    PHI = np.zeros([num_samples * num_dof, num_par])
    PHI_index_to = 0
    for sample_index in range(num_samples):
        PHI_index_from = PHI_index_to
        PHI_index_to = PHI_index_to + num_dof
        PHI_base = np.array(
            Hb.Hb(q_vect[sample_index, :], dq_vect[sample_index, :], ddq_vect[sample_index, :])
        ).reshape(num_dof, num_par - 2 * num_dof)
        PHI[PHI_index_from:PHI_index_to, :] = np.concatenate(
            [PHI_base, np.diag(dq_vect[sample_index, :]), np.diag(np.sign(dq_vect[sample_index, :]))], 1
        )
    return PHI


def get_M_vect(q_vect, w_base):
    """Computes the inertia matrices"""
    num_samples, num_dof = q_vect.shape
    num_par = w_base.size
    M = np.zeros([num_samples, num_dof, num_dof])
    phi_g = get_PHI(q_vect, np.zeros_like(q_vect), np.zeros_like(q_vect), num_par)
    g = np.matmul(phi_g, w_base).reshape([num_samples, num_dof])
    for joint_index in range(num_dof):
        # print('joint_index', joint_index)
        ddq_tmp = np.zeros_like(q_vect)
        ddq_tmp[:, joint_index] = 1.0
        phi_tmp = get_PHI(q_vect, np.zeros_like(q_vect), ddq_tmp, num_par)
        M[:, :, joint_index] = np.matmul(phi_tmp, w_base).reshape([num_samples, num_dof]) - g
        # tmp = np.matmul(phi_tmp, w_base).reshape([num_samples, num_dof])
        # print('M[0]', M[0])
        # print('tmp[0]', tmp[0])
        # print('g[0]', g[0])
    return M


def get_n_vect(q_vect, dq_vect, w_base):
    """Computes contributions independent on ddq"""
    num_samples, num_dof = q_vect.shape
    num_par = w_base.size
    phi_n = get_PHI(q_vect, dq_vect, np.zeros_like(q_vect), num_par)
    n = np.matmul(phi_n, w_base).reshape([num_samples, num_dof, 1])
    return n


# LOAD CONFIGURATION #
print("\n\n----- LOAD CONFIGURATION -----")

# read argparse parameters
p = argparse.ArgumentParser("Parametric estimator")
p.add_argument("-config_file", type=str, default="", help="Configuration file")
p.add_argument("-robot_name", type=str, default="", help="name of the robot, options: PANDA, MELFA")
d_argparse = vars(p.parse_known_args()[0])

# init config params
function_module = None
data_path = None
file_name_1 = None
file_name_2 = None
saving_path = None
num_dof = None
output_feature = None
noiseless_output_feature = None
downsampling_data_load = None
flg_norm = None
num_dat_tr = None
vel_threshold = None
num_par = None
flg_save = None
flg_plot = None


# read the config file
config = configparser.ConfigParser()
config.read(d_argparse["config_file"])
d_config = Config_Utils.get_d_param(config=config, kernel_name=d_argparse["robot_name"])

# load variables
locals().update(d_config)
robot_name = d_argparse["robot_name"]
if "flg_frict" not in d_config:
    flg_frict = False


# IMPORT FUNCTIONS #
print("\n\n----- IMPORT FUNCTIONS -----")
exec("import " + function_module + ".Hb as Hb")


# SET FILE NAME #
print("\n\n----- SET FILE NAME -----")

# loading path list
tr_path_list = [data_path + file_name for file_name in file_name_1.split(",")]
test1_path_list = [data_path + file_name for file_name in file_name_2.split(",")]

# saving and loading path
estimate_tr_saving_path = saving_path + "model_parametricID_" + robot_name + "_tr_estimates.pkl"
estimate_test1_saving_path = saving_path + "model_parametricID_" + robot_name + "_test1_estimates.pkl"
estimate_m_tr_saving_path = saving_path + "model_parametricID_" + robot_name + "_M_tr_estimates.pkl"
estimate_m_test1_saving_path = saving_path + "model_parametricID_" + robot_name + "_M_test1_estimates.pkl"


# LOAD DATA #
print("\n\n----- LOAD DATA -----")

# feature names
joint_index_list = range(0, num_dof)
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
q_names = ["q_" + joint_name for joint_name in joint_names]
dq_names = ["dq_" + joint_name for joint_name in joint_names]
ddq_names = ["ddq_" + joint_name for joint_name in joint_names]


# set features
input_features = q_names + dq_names + ddq_names
pos_indices = list(range(num_dof))
vel_indices = list(range(num_dof, num_dof * 2))
acc_indices = list(range(num_dof * 2, num_dof * 3))
num_dims = len(input_features)
input_features_joint_list = [input_features for i in range(num_dof)]

# training dataset
input_tr_list = []
output_tr_list = []
data_frame_tr_list = []
for tr_path in tr_path_list:
    input_tr, output_tr, active_dims_list, data_frame_tr = Project_Utils.get_data_from_features(
        tr_path, input_features, input_features_joint_list, output_feature, num_dof
    )
    input_tr_list.append(input_tr)
    output_tr_list.append(output_tr)
    data_frame_tr_list.append(data_frame_tr)
input_tr = np.concatenate(input_tr_list, 0)
output_tr = np.concatenate(output_tr_list, 0)
data_frame_tr = pd.concat(data_frame_tr_list, 0)
noiseless_output_tr = data_frame_tr[[noiseless_output_feature + "_" + str(i + 1) for i in joint_index_list]].values

# test dataset
input_test1_list = []
output_test1_list = []
data_frame_test1_list = []
for test1_path in test1_path_list:
    input_test1, output_test1, active_dims_list, data_frame_test1 = Project_Utils.get_data_from_features(
        test1_path, input_features, input_features_joint_list, output_feature, num_dof
    )
    noiseless_output_test1 = data_frame_test1[
        [noiseless_output_feature + "_" + str(i + 1) for i in joint_index_list]
    ].values
    input_test1_list.append(input_test1)
    output_test1_list.append(output_test1)
    data_frame_test1_list.append(data_frame_test1)
input_test1 = np.concatenate(input_test1_list, 0)
output_test1 = np.concatenate(output_test1_list, 0)
data_frame_test1 = pd.concat(data_frame_test1_list, 0)
noiseless_output_test1 = data_frame_test1[
    [noiseless_output_feature + "_" + str(i + 1) for i in joint_index_list]
].values


# downsampling
input_tr = input_tr[::downsampling_data_load, :]
output_tr = output_tr[::downsampling_data_load, :]
noiseless_output_tr = noiseless_output_tr[::downsampling_data_load, :]
data_frame_tr = data_frame_tr.iloc[::downsampling_data_load, :]
input_test1 = input_test1[::downsampling_data_load, :]
output_test1 = output_test1[::downsampling_data_load, :]
noiseless_output_test1 = noiseless_output_test1[::downsampling_data_load, :]
data_frame_test1 = data_frame_test1.iloc[::downsampling_data_load, :]

# normaliziation
if flg_norm:
    norm_coef = np.std(output_tr, 0)
    output_tr = output_tr / norm_coef
    noiseless_output_tr = noiseless_output_tr / norm_coef
    output_test1 = output_test1 / norm_coef
    noiseless_output_test1 = noiseless_output_test1 / norm_coef
    print("\nNormalization: ")
    print("norm_coef:", norm_coef)
else:
    norm_coef = [1.0] * num_dof
mean_coef = np.zeros(num_dof)
norm_coef_input = np.ones(num_dims)

# select the subset of training data to use
if num_dat_tr == -1:
    num_dat_tr = input_tr.shape[0]
indices_tr = np.arange(0, num_dat_tr)[
    np.sum(np.abs(data_frame_tr[dq_names].values[:num_dat_tr]) > vel_threshold, 1) == num_dof
]
print("num tr samples high vel", indices_tr.size)
input_tr = input_tr[indices_tr, :]
output_tr = output_tr[indices_tr, :]
noiseless_output_tr = noiseless_output_tr[indices_tr, :]
data_frame_tr = data_frame_tr.iloc[indices_tr, :]
num_dat_test1 = input_test1.shape[0]
num_dat_tr = indices_tr.size
print("input_tr.shape: ", input_tr.shape)
print("input_test1.shape: ", input_test1.shape)


# GET REGRESSION MATRIX #
print("\n\n----- GET REGRESSION MATRIX -----")
PHI_TR = get_PHI(
    data_frame_tr[q_names].values, data_frame_tr[dq_names].values, data_frame_tr[ddq_names].values, num_par
)
PHI_TEST1 = get_PHI(
    data_frame_test1[q_names].values, data_frame_test1[dq_names].values, data_frame_test1[ddq_names].values, num_par
)


# GET BASE PARAMETERS #
print("\n\n----- GET BASE PARAMETERS -----")
Y_TR = output_tr.reshape([num_dat_tr * num_dof, 1])
w_base = np.linalg.solve(np.matmul(PHI_TR.transpose(), PHI_TR), np.matmul(PHI_TR.transpose(), Y_TR))


# GET ESTIMATES #
print("\n\n----- GET ESTIMATES -----")
Y_TR_HAT = np.matmul(PHI_TR, w_base).reshape([num_dat_tr, num_dof])
Y_TEST1_HAT = np.matmul(PHI_TEST1, w_base).reshape([num_dat_test1, num_dof])
# save estimates
d_tr = Project_Utils.get_results_dict(
    Y=output_tr, Y_hat=Y_TR_HAT, norm_coef=norm_coef, mean_coef=mean_coef, Y_var=None, Y_noiseless=noiseless_output_tr
)
d_test1 = Project_Utils.get_results_dict(
    Y=output_test1,
    Y_hat=Y_TEST1_HAT,
    norm_coef=norm_coef,
    mean_coef=mean_coef,
    Y_var=None,
    Y_noiseless=noiseless_output_test1,
)
if flg_save:
    pkl.dump(d_tr, open(estimate_tr_saving_path, "wb"))
    pkl.dump(d_test1, open(estimate_test1_saving_path, "wb"))

# get the erros stats
print("\nnMSE")
Project_Utils.get_stat_estimate(Y=d_tr["Y_noiseless"], Y_hat=d_tr["Y_hat"], stat_name="nMSE")
Project_Utils.get_stat_estimate(Y=d_test1["Y_noiseless"], Y_hat=d_test1["Y_hat"], stat_name="nMSE")
print("\nMSE")
Project_Utils.get_stat_estimate(Y=d_tr["Y_noiseless"], Y_hat=d_tr["Y_hat"], stat_name="MSE")
Project_Utils.get_stat_estimate(Y=d_test1["Y_noiseless"], Y_hat=d_test1["Y_hat"], stat_name="MSE")

# print the estimates
if flg_plot:
    dq_tr = data_frame_tr[dq_names].values
    dq_test1 = data_frame_test1[dq_names].values
    Project_Utils.print_estimate_with_vel(
        Y=d_tr["Y"],
        Y_hat=d_tr["Y_hat"],
        joint_index_list=list(range(1, num_dof + 1)),
        dq=dq_tr,
        Y_noiseless=None,
        Y_hat_var=None,
        output_name="tau",
    )
    plt.show()
    Project_Utils.print_estimate_with_vel(
        Y=d_test1["Y"],
        Y_hat=d_test1["Y_hat"],
        joint_index_list=list(range(1, num_dof + 1)),
        dq=dq_test1,
        Y_noiseless=None,
        Y_hat_var=None,
        output_name="tau",
    )
    plt.show()
