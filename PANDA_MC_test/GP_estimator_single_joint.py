# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Script file for single joint estimation
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
        Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
"""
import sys

sys.path.insert(0, "../")

import argparse
import configparser
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import config_files.Utils as Config_Utils
import gpr_lib.Loss.Gaussian_likelihood as Likelihood
import Project_Utils_ as Project_Utils

# init config params
model_name = None
data_path = None
file_name_1 = None
file_name_2 = None
saving_path = None
model_loading_path = None
estimate_m_tr_saving_path = None
estimate_m_test1_saving_path = None
estimate_acc_tr_saving_path = None
estimate_acc_test1_saving_path = None
dev_name = None
num_threads = None
num_dof = None
output_feature = None
noiseless_output_feature = None
downsampling_data_load = None
flg_norm = None
num_dat_tr = None
vel_threshold = None
batch_size = None
robot_structure_str = None
model = None
sigma_n_num = None
flg_load = None
flg_train = None
lr = None
shuffle = None
n_epoch = None
n_epoch_print = None
flg_save = None
downsampling = None
file_name_m1 = None
file_name_m2 = None
single_sigma_n = None
loading_path = None
dyn_par_path = None
p_dyn_perc_error = None
p_dyn_add_std = None
pp_mean_function = None
pp_mean_function_module = None
pp_function_module = None
pp_mean_function_name = None

# ######### LOAD CONFIGURATION ##########
print("\n\n----- LOAD CONFIGURATION -----")

# read argparse parameters
p = argparse.ArgumentParser("GP estimator")
p.add_argument("-config_file", type=str, default="", help="Configuration file")
p.add_argument(
    "-kernel_name",
    type=str,
    default="",
    help="name of the kernel model, options:  m_ind_RBF, m_ind_LIN, m_ind_SP, m_ind_GIP, m_ind_GIP_with friction",
)
p.add_argument("-dev_name", type=str, default=None, help="if set overwrites dev selected in config file")
d_argparse = vars(p.parse_known_args()[0])

# read the config file
config = configparser.ConfigParser()
print("Reading parameters from ", d_argparse["config_file"])
config.read(d_argparse["config_file"])
d_config = Config_Utils.get_d_param(config=config, kernel_name=d_argparse["kernel_name"])

# load variables
locals().update(d_config)
# model_name = d_argparse['kernel_name']
kernel_name = d_argparse["kernel_name"]
if "flg_frict" not in d_config:
    flg_frict = False
if "flg_compute_acc" not in d_config:
    flg_compute_acc = False
if "flg_plot" not in d_config:
    flg_plot = False
if "num_par" not in d_config:
    num_par = num_dof * 10 + 2
if "drop_last" not in d_config:
    drop_last = False
if d_argparse["dev_name"] is not None:
    dev_name = d_argparse["dev_name"]


# ######### IMPORT FUNCTIONS ##########
print("\n\n----- IMPORT FUNCTIONS -----")
exec("from Models import " + model_name + " as model")


# ######### SET FILE NAME ##########
print("\n\n----- SET FILE NAME -----")

# loading path
tr_path = data_path + file_name_1
test1_path = data_path + file_name_2
M_tr_path = data_path + file_name_m1
M_test1_path = data_path + file_name_m2
print("tr_path:", tr_path)
print("test1_path:", test1_path)
print("M_tr_path:", M_tr_path)
print("M_test1_path:", M_test1_path)

# saving and loading path
model_saving_path = saving_path + "model_" + model_name + ".pt"
model_loading_path = model_loading_path + "model_" + model_name + ".pt"
# estimate_tr_saving_path = saving_path+'model_'+model_name+'_tr_estimates.pkl'
estimate_test1_saving_path = saving_path + "model_" + model_name + "_test1_estimates.pkl"
# estimate_m_tr_saving_path = saving_path+'model_'+model_name+'_M_tr_estimates.pkl'
# estimate_m_test1_saving_path = saving_path+'model_'+model_name+'_M_test1_estimates.pkl'
# estimate_acc_tr_saving_path = saving_path+'model_'+model_name+'_acc_tr_estimates.pkl'
# estimate_acc_test1_saving_path = saving_path+'model_'+model_name+'_acc_test1_estimates.pkl'


# ######### SET TYPE AND DEVICE ##########
print("\n\n----- SET TYPE AND DEVICE -----")

# set type
dtype = torch.float64

# set the device
device = torch.device(dev_name)
torch.set_num_threads(num_threads)


# ######### LOAD DATA ##########
print("\n\n----- LOAD DATA -----")

# feature names
joint_index_list = range(0, num_dof)
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
q_names = ["q_" + joint_name for joint_name in joint_names]
dq_names = ["dq_" + joint_name for joint_name in joint_names]
ddq_names = ["ddq_" + joint_name for joint_name in joint_names]


# get the robot structure
robot_structure = []
for c in range(len(robot_structure_str)):
    if c == "p":
        robot_structure.append(1)
    else:
        robot_structure.append(0)

if "GIP" in kernel_name:

    # training data
    input_tr, output_tr, active_dims_dict, data_frame_tr = Project_Utils.get_dataset_poly_from_structure(
        tr_path, num_dof, output_feature, robot_structure, features_name_list=joint_names
    )
    noiseless_output_tr = data_frame_tr[[noiseless_output_feature + "_" + str(i + 1) for i in joint_index_list]].values

    # test dataset
    input_test1, output_test1, active_dims_dict, data_frame_test1 = Project_Utils.get_dataset_poly_from_structure(
        test1_path, num_dof, output_feature, robot_structure, features_name_list=joint_names
    )
    noiseless_output_test1 = data_frame_test1[
        [noiseless_output_feature + "_" + str(i + 1) for i in joint_index_list]
    ].values

    # get pos and acc indices
    pos_indices = np.concatenate(active_dims_dict["active_dims_mon_rev"] + active_dims_dict["active_dims_mon_prism"])
    vel_indices = []
    acc_indices = active_dims_dict["active_dims_acc"]
    sin_indices = np.concatenate(active_dims_dict["active_dims_mon_rev_sin"])
    cos_indices = np.concatenate(active_dims_dict["active_dims_mon_rev_cos"])
    # prism_indices=np.concatenate(active_dims_dict['active_dims_mon_prism'])
    num_dims = input_tr.shape[1]
else:
    input_features = q_names + dq_names + ddq_names
    pos_indices = list(range(num_dof))
    vel_indices = list(range(num_dof, num_dof * 2))
    acc_indices = list(range(num_dof * 2, num_dof * 3))
    num_dims = len(input_features)
    input_features_joint_list = [input_features for i in range(num_dof)]

    # training dataset
    input_tr, output_tr, active_dims_list, data_frame_tr = Project_Utils.get_data_from_features(
        tr_path, input_features, input_features_joint_list, output_feature, num_dof
    )
    noiseless_output_tr = data_frame_tr[[noiseless_output_feature + "_" + str(i + 1) for i in joint_index_list]].values

    # test dataset
    input_test1, output_test1, active_dims_list, data_frame_test1 = Project_Utils.get_data_from_features(
        test1_path, input_features, input_features_joint_list, output_feature, num_dof
    )
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
print("input_tr.shape: ", input_tr.shape)
print("input_test1.shape: ", input_test1.shape)
if batch_size == -1 or batch_size is None:
    batch_size = num_dat_tr


# ######### GET THE MODEL ##########
print("\n\n----- GET THE MODEL -----")

# sigma_n init
if single_sigma_n:
    sigma_n_init = max([np.std(output_tr[:, i]) for i in range(0, num_dof)])
else:
    sigma_n_init = [np.std(output_tr[:, i]) for i in range(0, num_dof)]
flg_train_sigma_n = True

model_par_dict = {}

if kernel_name == "m_ind_RBF":
    model_par_dict["num_dof"] = num_dof
    model_par_dict["active_dims_list"] = active_dims_list
    model_par_dict["sigma_n_init_list"] = [np.std(output_tr[:, i]) for i in range(0, num_dof)]
    model_par_dict["flg_train_sigma_n"] = True
    model_par_dict["lengthscales_init_list"] = [np.ones(active_dims.size) for active_dims in active_dims_list]
    model_par_dict["flg_train_lengthscales_list"] = [True] * num_dof
    model_par_dict["lambda_init_list"] = [np.std(output_tr[:, i]) for i in range(0, num_dof)]
    # model_par_dict['lambda_init_list']=[np.ones(1) for i in range(0,num_dof)]
    model_par_dict["flg_train_lambda_list"] = [True] * num_dof
    model_par_dict["mean_init_list"] = None
    model_par_dict["flg_train_mean_list"] = False
    model_par_dict["pos_indices"] = pos_indices
    model_par_dict["acc_indices"] = acc_indices
    model_par_dict["norm_coef_input"] = norm_coef_input
    model_par_dict["name"] = kernel_name + "_"
    model_par_dict["dtype"] = dtype
    model_par_dict["max_input_loc"] = 10000
    model_par_dict["downsampling_mode"] = "Downsampling"
    model_par_dict["sigma_n_num"] = sigma_n_num
    model_par_dict["device"] = device
if kernel_name == "m_ind_SP_mean":
    model_par_dict["num_dof"] = num_dof
    model_par_dict["active_dims_list"] = active_dims_list
    model_par_dict["sigma_n_init_list"] = [np.std(output_tr[:, i]) for i in range(0, num_dof)]
    model_par_dict["flg_train_sigma_n"] = True
    model_par_dict["lengthscales_init_list"] = [np.ones(active_dims.size) for active_dims in active_dims_list]
    model_par_dict["flg_train_lengthscales_list"] = [True] * num_dof
    # model_par_dict['lambda_init_list']=[np.std(output_tr[:,i]) for i in range(0,num_dof)]
    model_par_dict["lambda_init_list"] = [np.ones(1) for i in range(0, num_dof)]
    model_par_dict["flg_train_lambda_list"] = [True] * num_dof
    model_par_dict["mean_init_list"] = None
    model_par_dict["flg_train_mean_list"] = False
    model_par_dict["pos_indices"] = pos_indices
    model_par_dict["acc_indices"] = acc_indices
    model_par_dict["norm_coef_input"] = norm_coef_input
    model_par_dict["name"] = kernel_name + "_"
    model_par_dict["dtype"] = dtype
    model_par_dict["max_input_loc"] = 10000
    model_par_dict["downsampling_mode"] = "Downsampling"
    model_par_dict["sigma_n_num"] = sigma_n_num
    model_par_dict["device"] = device
    # load nominal parameters
    p_dyn_nominal = pkl.load(open(dyn_par_path, "rb"))
    print("p_dyn_nominal", p_dyn_nominal)
    # add noise
    np.random.seed(0)
    num_p_dyn = p_dyn_nominal.size
    # multiplicative error (uniform in +-p_dyn_perc_error)
    p_dyn = p_dyn_nominal + p_dyn_perc_error * (2 * np.random.rand(num_p_dyn) - 1) * p_dyn_nominal
    # additive gaussian noise (with standard deviation p_dyn_add_std)
    p_dyn = p_dyn + p_dyn_add_std * np.random.randn(num_p_dyn)
    print("p_dyn", p_dyn)
    # parametric component params
    exec("from " + pp_mean_function_module + " import " + pp_mean_function_name + " as pp_mean_function")
    model_par_dict["f_mean_list"] = [pp_mean_function for i in range(0, num_dof)]
    model_par_dict["f_mean_add_par_dict_list"] = [
        {
            "pos_indices": pos_indices,
            "vel_indices": vel_indices,
            "acc_indices": acc_indices,
            "joint_index": i,
            "dtype": dtype,
            "device": device,
            "norm_coef": norm_coef[i],
        }
        for i in range(0, num_dof)
    ]
    model_par_dict["pos_par_mean_init_list"] = [[]] * num_dof
    model_par_dict["flg_train_pos_par_mean"] = [False] * num_dof
    model_par_dict["free_par_mean_init_list"] = [p_dyn for i in range(0, num_dof)]
    model_par_dict["flg_train_free_par_mean"] = [False for i in range(0, num_dof)]
elif kernel_name == "m_ind_LIN":  # parametric kernel
    # num_par = num_dof*10+2 # full par
    # import additional function
    import gpr_lib.Utils.Parameters_covariance_functions as cov_functions

    for joint_index in range(1, num_dof + 1):
        exec("import " + pp_function_module + ".H" + str(joint_index) + "_torch as H" + str(joint_index) + "_torch")
    # get PP basis functions
    f_transform_list = [
        eval("H" + str(joint_index) + "_torch.H" + str(joint_index)) for joint_index in range(1, num_dof + 1)
    ]
    print("f_transform_list", f_transform_list)
    # f_add_par_list = [[pos_indices, vel_indices, acc_indices, 0.00000001, dtype, device] for i in range(num_dof)]
    f_add_par_list = [[pos_indices, vel_indices, acc_indices, dtype, device] for i in range(num_dof)]
    model_par_dict["num_dof"] = num_dof
    model_par_dict["active_dims_list"] = [np.arange(num_par) for i in range(num_dof)]
    model_par_dict["f_transform"] = f_transform_list
    model_par_dict["f_add_par_list"] = f_add_par_list
    model_par_dict["sigma_n_init_list"] = [1 * np.ones(1)] * num_dof
    model_par_dict["flg_train_sigma_n"] = True
    model_par_dict["Sigma_function_list"] = [cov_functions.diagonal_covariance_ARD for i in range(0, num_dof)]
    model_par_dict["Sigma_f_additional_par_list"] = [[] for i in range(0, num_dof)]
    model_par_dict["Sigma_pos_par_init_list"] = [np.ones(num_par) for i in range(0, num_dof)]
    model_par_dict["flg_train_Sigma_pos_par"] = True
    model_par_dict["Sigma_free_par_init_list"] = [None for i in range(0, num_dof)]
    model_par_dict["flg_train_Sigma_free_par"] = False
    model_par_dict["pos_indices"] = pos_indices
    model_par_dict["acc_indices"] = acc_indices
    name = kernel_name + "_"
    model_par_dict["name"] = kernel_name + "_"
    model_par_dict["dtype"] = dtype
    model_par_dict["max_input_loc"] = 10000
    model_par_dict["downsampling_mode"] = "Downsampling"
    model_par_dict["sigma_n_num"] = sigma_n_num
    model_par_dict["device"] = device
elif kernel_name == "m_ind_SP":  # parametric kernel
    # num_par = num_dof*10+2 # full par
    # num_par = num_dof*10 # full par
    # import additional function
    import gpr_lib.Utils.Parameters_covariance_functions as cov_functions

    for joint_index in range(1, num_dof + 1):
        exec("import " + pp_function_module + ".H" + str(joint_index) + "_torch as H" + str(joint_index) + "_torch")
    # get PP basis functions
    f_transform_list = [
        eval("H" + str(joint_index) + "_torch.H" + str(joint_index)) for joint_index in range(1, num_dof + 1)
    ]
    # f_add_par_list = [[pos_indices, vel_indices, acc_indices, 0.00000001, dtype, device] for i in range(num_dof)]
    f_add_par_list = [[pos_indices, vel_indices, acc_indices, dtype, device] for i in range(num_dof)]
    model_par_dict["num_dof"] = num_dof
    model_par_dict["active_dims_list_PP"] = [np.arange(num_par) for i in range(num_dof)]
    model_par_dict["active_dims_list_RBF"] = [np.arange(3 * num_dof) for i in range(num_dof)]
    model_par_dict["f_transform"] = f_transform_list
    model_par_dict["f_add_par_list"] = f_add_par_list
    model_par_dict["sigma_n_init_list"] = [1 * np.ones(1)] * num_dof
    model_par_dict["flg_train_sigma_n"] = True
    model_par_dict["Sigma_function_list"] = [cov_functions.diagonal_covariance_ARD for i in range(0, num_dof)]
    model_par_dict["Sigma_f_additional_par_list"] = [[] for i in range(0, num_dof)]
    model_par_dict["Sigma_pos_par_init_list"] = [np.ones(num_par) for i in range(0, num_dof)]
    model_par_dict["flg_train_Sigma_pos_par"] = True
    model_par_dict["Sigma_free_par_init_list"] = [None for i in range(0, num_dof)]
    model_par_dict["flg_train_Sigma_free_par"] = False
    model_par_dict["lengthscales_init_list"] = [np.ones(3 * num_dof) for i in range(num_dof)]
    model_par_dict["flg_train_lengthscales_list"] = [True for i in range(num_dof)]
    model_par_dict["lambda_init_list"] = [np.ones(1) for i in range(num_dof)]
    model_par_dict["flg_train_lambda_list"] = [True for i in range(num_dof)]
    model_par_dict["pos_indices"] = pos_indices
    model_par_dict["acc_indices"] = acc_indices
    name = kernel_name + "_"
    model_par_dict["name"] = kernel_name + "_"
    model_par_dict["dtype"] = dtype
    model_par_dict["max_input_loc"] = 10000
    model_par_dict["downsampling_mode"] = "Downsampling"
    model_par_dict["sigma_n_num"] = sigma_n_num
    model_par_dict["device"] = device
elif "GIP" in kernel_name:
    # acc/vel kernel
    acc_vel_gp_dict = dict()
    acc_vel_gp_dict["name"] = "acc_vel"
    acc_vel_gp_dict["active_dims"] = [active_dims_dict["active_dims_acc_vel"]]
    acc_vel_gp_dict["poly_deg"] = [1]
    acc_vel_gp_dict["flg_offset"] = [True]
    acc_vel_gp_dict["sigma_n_init"] = [1 * np.ones(1)]
    acc_vel_gp_dict["flg_train_sigma_n"] = [True]
    acc_vel_gp_dict["Sigma_pos_par_init"] = [[1 * np.ones(active_dims_dict["active_dims_acc_vel"].size + 1)]]
    acc_vel_gp_dict["flg_train_Sigma_pos_par"] = [[True]]
    acc_vel_gp_dict["Sigma_free_par_init"] = [[None]]
    acc_vel_gp_dict["flg_train_Sigma_free_par"] = [[False]]
    acc_vel_gp_dict["sigma_n_num"] = [sigma_n_num]
    # position_kernel
    position_gp_dict = dict()
    position_gp_dict["name"] = "position_kernel"
    position_gp_dict["active_dims"] = (
        active_dims_dict["active_dims_mon_rev"] + active_dims_dict["active_dims_mon_prism"]
    )
    num_gp = len(position_gp_dict["active_dims"])
    position_gp_dict["poly_deg"] = [2] * num_gp
    position_gp_dict["flg_offset"] = [True] * num_gp
    position_gp_dict["sigma_n_init"] = [None] * num_gp
    position_gp_dict["flg_train_sigma_n"] = [False] * num_gp
    position_gp_dict["Sigma_pos_par_init"] = [
        [np.ones(position_gp_dict["active_dims"][i].size + 1), np.ones([2, position_gp_dict["active_dims"][i].size])]
        for i in range(num_gp)
    ]
    position_gp_dict["flg_train_Sigma_pos_par"] = [[True, True]] * num_gp
    position_gp_dict["Sigma_free_par_init"] = [[None, None]] * num_gp
    position_gp_dict["flg_train_Sigma_free_par"] = [[False, False]] * num_gp
    position_gp_dict["sigma_n_num"] = [None] * num_gp
    # move models to list
    gp_dict_list = [[acc_vel_gp_dict, position_gp_dict]] * num_dof
    model_par_dict["num_dof"] = num_dof
    model_par_dict["gp_dict_list"] = gp_dict_list
    model_par_dict["pos_indices"] = pos_indices
    model_par_dict["vel_indices"] = []
    model_par_dict["acc_indices"] = acc_indices
    model_par_dict["sin_indices"] = sin_indices
    model_par_dict["cos_indices"] = cos_indices
    model_par_dict["name"] = kernel_name + "_"
    model_par_dict["dtype"] = dtype
    model_par_dict["max_input_loc"] = 10000
    model_par_dict["downsampling_mode"] = "Downsampling"
    model_par_dict["sigma_n_num"] = sigma_n_num
    model_par_dict["device"] = device
    if kernel_name == "m_ind_GIP_with_friction":
        import gpr_lib.Utils.Parameters_covariance_functions as cov_functions

        gp_friction_dict_list = []
        for joint_index, active_dims_joint in enumerate(active_dims_dict["active_dims_friction"]):
            gp_friction_dict = dict()
            gp_friction_dict["name"] = "friction"
            gp_friction_dict["active_dims"] = active_dims_joint
            gp_friction_dict["Sigma_function"] = cov_functions.diagonal_covariance_ARD
            gp_friction_dict["Sigma_f_additional_par_list"] = []
            gp_friction_dict["Sigma_pos_par_init"] = np.ones(2)
            gp_friction_dict["Sigma_free_par_init"] = None
            gp_friction_dict["flg_train_Sigma_pos_par"] = True
            gp_friction_dict["flg_train_Sigma_free_par"] = False
            gp_friction_dict_list.append(gp_friction_dict)
        model_par_dict["gp_friction_dict_list"] = gp_friction_dict_list

# init the model
m = model(**model_par_dict)

# move the model to the device
m.to(device)
m.print_model()

# load the model
if flg_load:
    print("Load the model...")
    m.load_state_dict(torch.load(model_loading_path))
    m.print_model()


# ######### TRAIN/LOAD THE MODEL ##########
print("\n\n----- TRAIN/LOAD THE MODEL -----")

# set the number of cumputational threads
torch.set_num_threads(num_threads)

# load the model
if flg_load:
    print("Load the model...")
    m.load_state_dict(torch.load(model_loading_path))

# train the joint models minimizing the negative MLL
if flg_train:  # hyper optimization for inv dyns
    print("Train the model with minimizing the negative MLL...")
    f_optimizer = lambda p: torch.optim.Adam(p, lr=lr, weight_decay=0)
    m.train_model(
        joint_index_list=joint_index_list,
        X=input_tr,
        Y=output_tr,
        criterion=Likelihood.Marginal_log_likelihood(),
        f_optimizer=f_optimizer,
        batch_size=batch_size,
        shuffle=shuffle,
        N_epoch=n_epoch,
        N_epoch_print=n_epoch_print,
        p_drop=0.0,
        drop_last=drop_last,
    )

# save the model
if flg_save:
    print("\nSaving model to " + model_saving_path + "\n")
    m.cpu()
    torch.save(m.state_dict(), model_saving_path)
    m.to(device)


# ######### GET THE ESTIMATE ##########
print("\n\n----- GET TRAINING AND TEST ESTIMATES -----")

with torch.no_grad():
    print("Training estimate...")
    t_start = time.time()
    Y_tr_hat_list, var_tr_list, alpha_tr_list, m_X_list, K_X_inv_list, _ = m.get_torque_estimates(
        input_tr[::downsampling],
        output_tr[::downsampling],
        input_tr,
        joint_indices_list=joint_index_list,
        flg_return_K_X_inv=True,
    )
    t_stop = time.time()
    print("Time elapsed ", t_stop - t_start)

    print("Test1 estimate...")
    t_start = time.time()
    Y_test1_hat_list, var_test1_list, _, _, _, _ = m.get_torque_estimates(
        input_tr[::downsampling],
        output_tr[::downsampling],
        input_test1,
        alpha_list_par=alpha_tr_list,
        m_X_list_par=m_X_list,
        joint_indices_list=joint_index_list,
        K_X_inv_list_par=K_X_inv_list,
    )
    t_stop = time.time()
    print("Time elapsed ", t_stop - t_start)

# get results dict (denormalized signals)
d_tr = Project_Utils.get_results_dict(
    Y=output_tr,
    Y_hat=np.concatenate(Y_tr_hat_list, 1),
    norm_coef=norm_coef,
    mean_coef=mean_coef,
    Y_var=np.concatenate(var_tr_list, 1),
    Y_noiseless=noiseless_output_tr,
)
d_test1 = Project_Utils.get_results_dict(
    Y=output_test1,
    Y_hat=np.concatenate(Y_test1_hat_list, 1),
    norm_coef=norm_coef,
    mean_coef=mean_coef,
    Y_var=np.concatenate(var_test1_list, 1),
    Y_noiseless=noiseless_output_test1,
)

# save results
if flg_save:
    # pkl.dump(d_tr, open(estimate_tr_saving_path, 'wb'))
    pkl.dump([d_tr, d_test1], open(estimate_test1_saving_path, "wb"))

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
        Y_hat_var=d_tr["Y_var"],
        output_name="tau",
    )
    plt.show()
    Project_Utils.print_estimate_with_vel(
        Y=d_test1["Y"],
        Y_hat=d_test1["Y_hat"],
        joint_index_list=list(range(1, num_dof + 1)),
        dq=dq_test1,
        Y_noiseless=None,
        Y_hat_var=d_test1["Y_var"],
        output_name="tau",
    )
    plt.show()


if flg_compute_acc:
    # ######### GET ACCELERATION ESTIMATES ##########
    print("\n\n----- COMPUTE ACCELERATION ESTIMATES -----")

    print("\nCompute the estimated inertia matrices...")
    # load inertia matrix
    M_tr = pkl.load(open(M_tr_path, "rb"))
    M_test1 = pkl.load(open(M_test1_path, "rb"))
    M_tr = np.stack(M_tr[::downsampling_data_load], axis=0)[:num_dat_tr]
    M_test1 = np.stack(M_test1[::downsampling_data_load], axis=0)

    # compute inertia matrix estimates
    with torch.no_grad():
        # compute the matrices
        M_tr_hat = m.get_M_estimates(
            X_tr=torch.tensor(input_tr[::downsampling], device=device, dtype=dtype),
            X_test=torch.tensor(input_tr, device=device, dtype=dtype),
            alpha_par_list=alpha_tr_list,
            norm_coef=norm_coef,
        )
        M_test1_hat = m.get_M_estimates(
            X_tr=torch.tensor(input_tr[::downsampling], device=device, dtype=dtype),
            X_test=torch.tensor(input_test1, device=device, dtype=dtype),
            alpha_par_list=alpha_tr_list,
            norm_coef=norm_coef,
        )
        # move the matrices to numpy
        M_tr_hat = M_tr_hat.detach().cpu().numpy()
        M_test1_hat = M_test1_hat.detach().cpu().numpy()

    # mean square error
    err_M_tr = np.mean(np.sqrt((M_tr - M_tr_hat) ** 2), axis=(1, 2))
    print("\nTraining mean MSE M: ", np.mean(err_M_tr))
    err_M_test1 = np.mean(np.sqrt((M_test1 - M_test1_hat) ** 2), axis=(1, 2))
    print("Test mean MSE M: ", np.mean(err_M_test1))

    # check positivity
    eig_tr, _ = np.linalg.eig(M_tr)
    eig_tr_hat, _ = np.linalg.eig(M_tr_hat)
    eig_test1, _ = np.linalg.eig(M_test1)
    eig_test1_hat, _ = np.linalg.eig(M_test1_hat)
    non_positive_def_count_tr = np.sum(np.min(eig_tr_hat, 1) <= 0)
    non_positive_def_count_test1 = np.sum(np.min(eig_test1_hat, 1) <= 0)
    print("Number of training samples with non-positive inertia: ", non_positive_def_count_tr)
    print("Number of test samples with non-positive inertia: ", non_positive_def_count_test1)

    if flg_save:
        pkl.dump({"M_tr": M_tr, "M_tr_hat": M_tr_hat}, open(estimate_m_tr_saving_path, "wb"))
        pkl.dump({"M_test1": M_test1, "M_test_hat": M_test1_hat}, open(estimate_m_test1_saving_path, "wb"))

    # Compute accelerations
    print("\nCompute accelerations...")

    # extract the joint velocities and acc
    q_dot_names = ["dq_" + str(i + 1) for i in range(0, num_dof)]
    q_dot_tr = data_frame_tr[q_dot_names].values
    q_dot_test1 = data_frame_test1[q_dot_names].values
    q_ddot_names = ["ddq_" + str(i + 1) for i in range(0, num_dof)]
    q_ddot_tr = data_frame_tr[q_ddot_names].values[:num_dat_tr:, :]
    q_ddot_test1 = data_frame_test1[q_ddot_names].values

    # get cg estimataes
    with torch.no_grad():
        # compute the matrices
        cg_tr_hat = m.get_cg_estimates(
            X_tr=torch.tensor(input_tr[::downsampling], device=device, dtype=dtype),
            X_test=torch.tensor(input_tr, device=device, dtype=dtype),
            alpha_par_list=alpha_tr_list,
            norm_coef=norm_coef,
        )
        cg_test1_hat = m.get_cg_estimates(
            X_tr=torch.tensor(input_tr[::downsampling], device=device, dtype=dtype),
            X_test=torch.tensor(input_test1, device=device, dtype=dtype),
            alpha_par_list=alpha_tr_list,
            norm_coef=norm_coef,
        )
        # move the matrices to numpy
        cg_tr_hat = cg_tr_hat.detach().cpu().numpy()
        cg_test1_hat = cg_test1_hat.detach().cpu().numpy()

    # get the estimated acc
    q_ddot_tr_hat = np.zeros([num_dat_tr, num_dof])
    for sample_index in range(0, num_dat_tr):
        q_ddot_tr_hat[sample_index, :] = np.matmul(
            np.linalg.inv(M_tr_hat[sample_index, :, :]),
            -cg_tr_hat[sample_index, :].reshape([num_dof, 1])
            + (norm_coef * output_tr[sample_index, :]).reshape([num_dof, 1]),
        ).squeeze()
    q_ddot_test1_hat = np.zeros([num_dat_test1, num_dof])
    for sample_index in range(0, num_dat_test1):
        q_ddot_test1_hat[sample_index, :] = np.matmul(
            np.linalg.inv(M_test1_hat[sample_index, :, :]),
            -cg_test1_hat[sample_index, :].reshape([num_dof, 1])
            + (norm_coef * output_test1[sample_index, :]).reshape([num_dof, 1]),
        ).squeeze()

    # print mean errors
    print("\nTraining nMSE on accelerations:")
    Project_Utils.get_stat_estimate(Y=q_ddot_tr, Y_hat=q_ddot_tr_hat, stat_name="nMSE")
    print("Test nMSE on accelerations:")
    Project_Utils.get_stat_estimate(Y=q_ddot_test1, Y_hat=q_ddot_test1_hat, stat_name="nMSE")
    print("\nTraining MSE on accelerations:")
    Project_Utils.get_stat_estimate(Y=q_ddot_tr, Y_hat=q_ddot_tr_hat, stat_name="MSE")
    print("\nTest MSE on accelerations:")
    Project_Utils.get_stat_estimate(Y=q_ddot_test1, Y_hat=q_ddot_test1_hat, stat_name="MSE")

    # save estimates
    if flg_save:
        pkl.dump({"q_ddot_tr": q_ddot_tr, "q_ddot_tr_hat": q_ddot_tr_hat}, open(estimate_acc_tr_saving_path, "wb"))
        pkl.dump(
            {"q_ddot_test": q_ddot_test1, "q_ddot_test_hat": q_ddot_test1_hat},
            open(estimate_acc_test1_saving_path, "wb"),
        )

    # print acc estimates
    if flg_plot:
        Project_Utils.print_estimate(
            Y=q_ddot_tr,
            Y_hat=q_ddot_tr_hat,
            joint_index_list=list(range(1, num_dof + 1)),
            Y_noiseless=None,
            Y_hat_var=None,
            output_name="acc",
        )
        plt.show()
        Project_Utils.print_estimate(
            Y=q_ddot_test1,
            Y_hat=q_ddot_test1_hat,
            joint_index_list=list(range(1, num_dof + 1)),
            Y_noiseless=None,
            Y_hat_var=None,
            output_name="acc",
        )
        plt.show()
