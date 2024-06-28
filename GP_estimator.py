# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Script file for LK estimation
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
        Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
        Diego Romeres (romeres@merl.com)
"""

import argparse
import configparser
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import config_files.Utils as Config_Utils
import gpr_lib.Loss.Gaussian_likelihood as Likelihood
import Models_Lagrangian_kernel
import Project_Utils_ as Project_Utils

# LOAD CONFIGURATION #
print("\n\n----- LOAD CONFIGURATION -----")

# read argparse parameters
p = argparse.ArgumentParser("LK estimator")
p.add_argument("-config_file", type=str, default="", help="Configuration file")
p.add_argument(
    "-kernel_name", type=str, default="", help="name of the kernel model, options: GIP, GIP_vel, POLYRBF, RBF, RBF_M"
)
d_argparse = vars(p.parse_known_args()[0])

# init config params
lk_model_name = None
f_k_name = None
data_path = None
file_name_1 = None
file_name_2 = None
saving_path = None
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
num_prism = None
num_rev = None
robot_structure_str = None
model = None
f_K_blocks = None
f_K_blocks_ltr = None
f_K_blocks_diag = None
f_K_L_Y_blocks = None
f_K_T_Y_blocks = None
f_K_U_Y_blocks = None
sigma_n_num = None
flg_norm_noise = None
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


# read the config file
config = configparser.ConfigParser()
config.read(d_argparse["config_file"])
d_config = Config_Utils.get_d_param(config=config, kernel_name=d_argparse["kernel_name"])

# load variables
locals().update(d_config)
model_name = d_argparse["kernel_name"]
if "flg_frict" not in d_config:
    flg_frict = False
if "flg_np" not in d_config:
    flg_np = False
if "drop_last" not in d_config:
    drop_last = False

# IMPORT FUNCTIONS #
print("\n\n----- IMPORT FUNCTIONS -----")
exec("from Models_Lagrangian_kernel import " + lk_model_name + " as model")
exec("from gpr_lib.GP_prior.LK." + f_k_name + "_torch import " + f_k_name + " as f_K_blocks")
exec("from gpr_lib.GP_prior.LK." + f_k_name + "_ltr_torch import " + f_k_name + " as f_K_blocks_ltr")
exec("from gpr_lib.GP_prior.LK." + f_k_name + "_diag_torch import " + f_k_name + " as f_K_blocks_diag")
exec("from gpr_lib.GP_prior.LK." + f_k_name + "_L_Y_cov_torch import " + f_k_name + " as f_K_L_Y_blocks")
exec("from gpr_lib.GP_prior.LK." + f_k_name + "_T_Y_cov_torch import " + f_k_name + " as f_K_T_Y_blocks")
exec("from gpr_lib.GP_prior.LK." + f_k_name + "_U_Y_cov_torch import " + f_k_name + " as f_K_U_Y_blocks")


# SET FILE NAME #
print("\n\n----- SET FILE NAME -----")

# loading path
tr_path = data_path + file_name_1
test1_path = data_path + file_name_2
M_tr_path = data_path + file_name_m1
M_test1_path = data_path + file_name_m2

# saving and loading path
model_saving_path = saving_path + "model_" + model_name + ".pt"
model_loading_path = saving_path + "model_" + model_name + ".pt"
estimate_tr_saving_path = saving_path + "model_" + model_name + "_tr_estimates.pkl"
estimate_test1_saving_path = saving_path + "model_" + model_name + "_test1_estimates.pkl"
estimate_m_tr_saving_path = saving_path + "model_" + model_name + "_M_tr_estimates.pkl"
estimate_m_test1_saving_path = saving_path + "model_" + model_name + "_M_test1_estimates.pkl"
estimate_acc_tr_saving_path = saving_path + "model_" + model_name + "_acc_tr_estimates.pkl"
estimate_acc_test1_saving_path = saving_path + "model_" + model_name + "_acc_test1_estimates.pkl"


# SET TYPE AND DEVICE #
print("\n\n----- SET TYPE AND DEVICE -----")

# set type
dtype = torch.float64

# set the device
device = torch.device(dev_name)
torch.set_num_threads(num_threads)


# LOAD DATA #
print("\n\n----- LOAD DATA -----")

# set joint names and features
joint_index_list = range(0, num_dof)
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
q_names = ["q_" + joint_name for joint_name in joint_names]
dq_names = ["dq_" + joint_name for joint_name in joint_names]
ddq_names = ["ddq_" + joint_name for joint_name in joint_names]
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


# GET THE MODEL #
print("\n\n----- GET THE MODEL -----")

# sigma_n init
if single_sigma_n:
    sigma_n_init = max([np.std(output_tr[:, i]) for i in range(0, num_dof)])
else:
    sigma_n_init = [np.std(output_tr[:, i]) for i in range(0, num_dof)]
init_param_dict = {}

# parameters init
if model_name == "LK_GIP_vel":
    # GIP vel model
    init_param_dict["sigma_kin_vel_init"] = np.ones([int((num_dof**2 - num_dof) / 2 + num_dof)])
    init_param_dict["flg_train_sigma_kin_vel"] = True
    if num_prism > 0:
        init_param_dict["sigma_kin_pos_prism_init"] = np.ones([num_prism, 2])
        init_param_dict["flg_train_sigma_kin_pos_prism"] = True
        init_param_dict["sigma_pot_prism_init"] = np.ones([num_prism, 2])
        init_param_dict["flg_train_sigma_pot_prism"] = True
    if num_rev > 0:
        init_param_dict["sigma_kin_pos_rev_init"] = np.ones([num_rev, 3])
        init_param_dict["flg_train_sigma_kin_pos_rev"] = True
        init_param_dict["sigma_pot_rev_init"] = np.ones([num_rev, 3])
        init_param_dict["flg_train_sigma_pot_rev"] = True
elif model_name == "LK_GIP":
    # GIP model
    init_param_dict["sigma_kin_vel_init"] = np.ones(num_dof)
    init_param_dict["flg_train_sigma_kin_vel"] = True
    if num_prism > 0:
        init_param_dict["sigma_kin_pos_prism_init"] = np.ones([num_prism, 2])
        init_param_dict["flg_train_sigma_kin_pos_prism"] = True
        init_param_dict["sigma_pot_prism_init"] = np.ones([num_prism, 2])
        init_param_dict["flg_train_sigma_pot_prism"] = True
    if num_rev > 0:
        init_param_dict["sigma_kin_pos_rev_init"] = np.ones([num_rev, 3])
        init_param_dict["flg_train_sigma_kin_pos_rev"] = True
        init_param_dict["sigma_pot_rev_init"] = np.ones([num_rev, 3])
        init_param_dict["flg_train_sigma_pot_rev"] = True
elif model_name == "LK_POLY_RBF":
    init_param_dict["lengthscales_T_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_T"] = True
    init_param_dict["scale_T_init"] = np.ones(1)
    init_param_dict["flg_train_scale_T"] = True
    init_param_dict["sigma_POLY_init"] = np.ones(num_dof)
    init_param_dict["flg_train_sigma_POLY"] = True
    init_param_dict["lengthscales_U_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_U"] = True
    init_param_dict["scale_U_init"] = np.ones(1)
    init_param_dict["flg_train_scale_U"] = True
elif model_name == "LK_POLY_vel_RBF":
    init_param_dict["lengthscales_T_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_T"] = True
    init_param_dict["scale_T_init"] = np.ones(1)
    init_param_dict["flg_train_scale_T"] = True
    init_param_dict["sigma_POLY_init"] = np.ones(int((num_dof**2 - num_dof) / 2 + num_dof))
    init_param_dict["flg_train_sigma_POLY"] = True
    init_param_dict["lengthscales_U_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_U"] = True
    init_param_dict["scale_U_init"] = np.ones(1)
    init_param_dict["flg_train_scale_U"] = True
elif model_name == "LK_GIP_sum":
    # get the robot structure
    robot_structure = []
    for c in range(len(robot_structure_str)):
        if c == "p":
            robot_structure.append(1)
        else:
            robot_structure.append(0)
    init_param_dict["sigma_kin_vel_list_init"] = [np.ones(joint_index_1) for joint_index_1 in range(1, num_dof + 1)]
    init_param_dict["flg_train_sigma_kin_vel"] = True
    sigma_kin_pos_list = []
    sigma_pot_list = []
    for joint_index_1 in range(num_dof):
        for joint_index_2 in range(joint_index_1 + 1):
            if robot_structure[joint_index_2] == 1:
                sigma_kin_pos_list.append(np.ones(2))
            else:
                sigma_kin_pos_list.append(np.ones(3))
        if robot_structure[joint_index_1] == 1:
            sigma_pot_list.append(np.ones(2))
        else:
            sigma_pot_list.append(np.ones(3))
    init_param_dict["sigma_kin_pos_list_init"] = sigma_kin_pos_list
    init_param_dict["flg_train_sigma_kin_pos"] = True
    init_param_dict["sigma_pot_list_init"] = sigma_pot_list
    init_param_dict["flg_train_sigma_pot"] = True
elif model_name == "LK_RBF":
    # RBF model
    init_param_dict["lengthscales_T_init"] = np.ones(num_dof * 2)
    init_param_dict["flg_train_lengthscales_T"] = True
    init_param_dict["scale_T_init"] = np.ones(1)
    init_param_dict["flg_train_scale_T"] = True
    init_param_dict["lengthscales_U_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_U"] = True
    init_param_dict["scale_U_init"] = np.ones(1)
    init_param_dict["flg_train_scale_U"] = True
elif model_name == "LK_RBF_1":
    # RBF model
    init_param_dict["lengthscales_L_init"] = np.ones(num_dof * 2)
    init_param_dict["flg_train_lengthscales_L"] = True
    init_param_dict["scale_L_init"] = np.ones(1)
    init_param_dict["flg_train_scale_L"] = True
elif model_name == "LK_RBF_1_sc":
    # RBF model
    init_param_dict["lengthscales_L_init"] = np.ones(num_dof * 3)
    init_param_dict["flg_train_lengthscales_L"] = True
    init_param_dict["scale_L_init"] = np.ones(1)
    init_param_dict["flg_train_scale_L"] = True
elif model_name == "LK_RBF_M":
    # RBF model
    init_param_dict["lengthscales_T_init_list"] = [
        np.ones(num_dof) for joint_index_1 in range(num_dof) for joint_index_2 in range(joint_index_1 + 1)
    ]
    init_param_dict["flg_train_lengthscales_T"] = True
    init_param_dict["scale_T_init_list"] = [
        np.ones(1) for joint_index_1 in range(num_dof) for joint_index_2 in range(joint_index_1 + 1)
    ]
    init_param_dict["flg_train_scale_T"] = True
    init_param_dict["lengthscales_U_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_U"] = True
    init_param_dict["scale_U_init"] = np.ones(1)
    init_param_dict["flg_train_scale_U"] = True
elif model_name == "LK_POLY_RBF_sum":
    # RBF sum model
    init_param_dict["sigma_kin_vel_list_init"] = [np.ones(joint_index) for joint_index in range(1, num_dof + 1)]
    init_param_dict["flg_train_sigma_kin_vel"] = True
    init_param_dict["lengthscales_T_list_init"] = [np.ones(joint_index) for joint_index in range(1, num_dof + 1)]
    init_param_dict["flg_train_lengthscales_T"] = True
    init_param_dict["scale_T_init"] = np.ones(num_dof)
    init_param_dict["flg_train_scale_T"] = True
    init_param_dict["lengthscales_U_init"] = np.ones(num_dof)
    init_param_dict["flg_train_lengthscales_U"] = True
    init_param_dict["scale_U_init"] = np.ones(1)
    init_param_dict["flg_train_scale_U"] = True
# check friction
if flg_frict:
    if "friction_model" not in d_config:
        friction_model = "linear"
    if friction_model == "linear":
        f_phi_friction = Models_Lagrangian_kernel.f_phi_friction_basic
        init_param_dict["K_friction_par_init"] = np.ones([num_dof, 2])
        init_param_dict["flg_train_friction_par"] = True

    elif friction_model == "RBF":
        f_phi_friction = Models_Lagrangian_kernel.f_phi_friction_basic
        init_param_dict["scale_friction_par_init"] = np.ones([num_dof])
        init_param_dict["lengthscale_friction_par_init"] = np.ones([num_dof, 2])
        init_param_dict["flg_train_friction_par"] = True
else:
    f_phi_friction = None
    friction_model = None
# check non parametric flag
if flg_np:
    init_param_dict["scale_NP_par_init"] = np.ones([num_dof])
    init_param_dict["lengthscale_NP_par_init"] = np.ones([num_dof, num_dims])
    init_param_dict["flg_train_NP_par"] = True

# init the model
m = model(
    num_dof=num_dof,
    pos_indices=pos_indices,
    vel_indices=vel_indices,
    acc_indices=acc_indices,
    init_param_dict=init_param_dict,
    f_K_blocks=f_K_blocks,
    f_K_blocks_ltr=f_K_blocks_ltr,
    f_K_blocks_diag=f_K_blocks_diag,
    f_K_L_Y_blocks=f_K_L_Y_blocks,
    f_K_T_Y_blocks=f_K_T_Y_blocks,
    f_K_U_Y_blocks=f_K_U_Y_blocks,
    friction_model=friction_model,
    f_phi_friction=f_phi_friction,
    flg_np=flg_np,
    sigma_n_init=sigma_n_init,
    flg_train_sigma_n=True,
    name=model_name,
    dtype=dtype,
    device=device,
    sigma_n_num=sigma_n_num,
    norm_coef=norm_coef,
    flg_norm_noise=flg_norm_noise,
)

# move the model to the device
m.to(device)

# load the model
if flg_load:
    print("Load the model...")
    m.load_state_dict(torch.load(model_loading_path))
    m.print_model()


# TRAIN THE MODEL #

# train the model
if flg_train:
    print("\n----- TRAIN THE MODEL -----")
    f_optimizer = lambda p: torch.optim.Adam(p, lr=lr, weight_decay=0)
    print("\nTrain the model minimizing the negative MLL...")
    m.train_model(
        X=input_tr,
        Y=output_tr,
        criterion=Likelihood.Marginal_log_likelihood(),
        f_optimizer=f_optimizer,
        batch_size=batch_size,
        shuffle=shuffle,
        N_epoch=n_epoch,
        N_epoch_print=n_epoch_print,
        drop_last=drop_last,
    )

    # save the model
    if flg_save:
        print("\nSave the model...")
        m.cpu()
        torch.save(m.state_dict(), model_saving_path)
        m.to(device)


# GET TORQUES ESTIMATES #
print("\n----- GET TORQUES ESTIMATES -----")

# estimate torques
with torch.no_grad():
    # training
    print("\nTraining estimate...")
    t_start = time.time()
    Y_tr_hat, var_tr, alpha_tr, K_X_inv = m.get_torques_estimate(
        X=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
        Y=torch.tensor(output_tr[::downsampling], dtype=dtype, device=device),
        X_test=torch.tensor(input_tr, dtype=dtype, device=device),
        flg_return_K_X_inv=True,
    )
    Y_tr_hat = Y_tr_hat.detach().cpu().numpy()
    var_tr = var_tr.detach().cpu().numpy()
    t_stop = time.time()
    print("Time elapsed ", t_stop - t_start)
    # test
    print("\nTest1 estimate...")
    t_start = time.time()
    Y_test1_hat, var_test1 = m.get_estimate_from_alpha(
        X=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
        X_test=torch.tensor(input_test1, dtype=dtype, device=device),
        alpha=alpha_tr,
        K_X_inv=K_X_inv,
    )
    Y_test1_hat = Y_test1_hat.detach().cpu().numpy()
    var_test1 = var_test1.detach().cpu().numpy()
    t_stop = time.time()
    print("Time elapsed ", t_stop - t_start)

# denormalize signals and move results to dictionary
d_tr = Project_Utils.get_results_dict(
    Y=output_tr, Y_hat=Y_tr_hat, norm_coef=norm_coef, mean_coef=mean_coef, Y_var=var_tr, Y_noiseless=noiseless_output_tr
)
d_test1 = Project_Utils.get_results_dict(
    Y=output_test1,
    Y_hat=Y_test1_hat,
    norm_coef=norm_coef,
    mean_coef=mean_coef,
    Y_var=var_test1,
    Y_noiseless=noiseless_output_test1,
)
if flg_save:
    pkl.dump(d_tr, open(estimate_tr_saving_path, "wb"))
    pkl.dump([d_tr, d_test1], open(estimate_test1_saving_path, "wb"))

# get the erros stats
print("\nnMSE")
Project_Utils.get_stat_estimate(Y=d_tr["Y_noiseless"], Y_hat=d_tr["Y_hat"], stat_name="nMSE")
Project_Utils.get_stat_estimate(Y=d_test1["Y_noiseless"], Y_hat=d_test1["Y_hat"], stat_name="nMSE")

print("\nMSE")
Project_Utils.get_stat_estimate(Y=d_tr["Y_noiseless"], Y_hat=d_tr["Y_hat"], stat_name="MSE")
Project_Utils.get_stat_estimate(Y=d_test1["Y_noiseless"], Y_hat=d_test1["Y_hat"], stat_name="MSE")

# print the estimates
dq_tr = data_frame_tr[dq_names].values
dq_test1 = data_frame_test1[dq_names].values
Project_Utils.print_estimate_single_fig(
    Y=d_tr["Y"],
    Y_hat=d_tr["Y_hat"],
    joint_index_list=[i + 1 for i in range(num_dof)],
    # dq=dq_tr,
    Y_noiseless=d_tr["Y_noiseless"],
    Y_hat_var=d_tr["Y_var"],
    output_name="tau",
    title="Torque estimates on training trajectory",
)
# plt.show()
Project_Utils.print_estimate_single_fig(
    Y=d_test1["Y"],
    Y_hat=d_test1["Y_hat"],
    joint_index_list=[i + 1 for i in range(num_dof)],
    # dq=dq_test1
    Y_noiseless=None,
    Y_hat_var=d_test1["Y_var"],
    output_name="tau",
    title="Torque estimates on test trajectory",
)
plt.show()


# Get rank of the K matrix
print("\n\n----- GET RANK OF K -----")
with torch.no_grad():
    K = m.get_K(
        torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
        torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
        flg_frict=flg_frict,
        flg_np=flg_np,
    )
    rk_K = np.linalg.matrix_rank(K.detach().cpu().numpy())

print("\n K has dimensions: {} x {}".format(K.shape[0], K.shape[1]))
print("\n K has rank: {}".format(rk_K))


if "RBF_1" in model_name:
    quit()

# GET INERTIA MATRIX ESTIMATES #
print("\n\n----- GET INERTIA MATRIX ESTIMATES -----")

# load inertia matrix
M_tr = pkl.load(open(M_tr_path, "rb"))
M_test1 = pkl.load(open(M_test1_path, "rb"))
M_tr = np.stack(M_tr[::downsampling_data_load], axis=0)[:num_dat_tr]
M_test1 = np.stack(M_test1[::downsampling_data_load], axis=0)

# compute inertia matrix estimates
with torch.no_grad():
    if "RBF_1" in model_name:
        # training
        print("\nTraining Inertia Matrices...")
        t_start = time.time()
        M_tr_hat_ = (
            m.get_M_estimates(
                X_tr=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
                X_test=torch.tensor(input_tr, dtype=dtype, device=device),
                alpha=alpha_tr,
            )
            .detach()
            .cpu()
            .numpy()
        )
        t_stop = time.time()
        print("Time elapsed ", t_stop - t_start)
        # test
        print("\nTest Inertia Matrices...")
        t_start = time.time()
        M_test1_hat_ = (
            m.get_M_estimates(
                X_tr=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
                X_test=torch.tensor(input_test1, dtype=dtype, device=device),
                alpha=alpha_tr,
            )
            .detach()
            .cpu()
            .numpy()
        )
        t_stop = time.time()
        print("Time elapsed ", t_stop - t_start)
    else:
        # training
        print("\nTraining Inertia Matrices from kin energy...")
        t_start = time.time()
        M_tr_hat_ = (
            m.get_M_estimates_T(
                X_tr=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
                X_test=torch.tensor(input_tr, dtype=dtype, device=device),
                alpha=alpha_tr,
            )
            .detach()
            .cpu()
            .numpy()
        )
        t_stop = time.time()
        print("Time elapsed ", t_stop - t_start)
        # test
        print("\nTest Inertia Matrices from kin energy...")
        t_start = time.time()
        M_test1_hat_ = (
            m.get_M_estimates_T(
                X_tr=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device),
                X_test=torch.tensor(input_test1, dtype=dtype, device=device),
                alpha=alpha_tr,
            )
            .detach()
            .cpu()
            .numpy()
        )
        t_stop = time.time()
        print("Time elapsed ", t_stop - t_start)


# mean square error
# err_M_tr = np.mean(np.sqrt((M_tr-M_tr_hat)**2), axis=(1,2))
err_M_tr_ = np.mean(np.sqrt((M_tr - M_tr_hat_) ** 2), axis=(1, 2))
# print('\nTraining mean MSE: ',np.mean(err_M_tr))
print("Training mean MSE kin: ", np.mean(err_M_tr_))
# err_M_test1 = np.mean(np.sqrt((M_test1-M_test1_hat)**2), axis=(1,2))
err_M_test1_ = np.mean(np.sqrt((M_test1 - M_test1_hat_) ** 2), axis=(1, 2))
# print('\nTest mean MSE: ',np.mean(err_M_test1))
print("Test mean MSE kin: ", np.mean(err_M_test1_))

# check simmetry
symm_err_tr = M_tr_hat_ - np.transpose(M_tr_hat_, axes=(0, 2, 1))
simm_mismatches_tr = np.sum(
    [np.allclose(M_tr_hat_[i, :, :], M_tr_hat_[i, :, :].T, rtol=0.0, atol=1e-5) for i in range(M_tr_hat_.shape[0])]
)
print(
    "\nNumber of training samples with non-symmetric inertia (tolerance 1e-5): ",
    M_tr_hat_.shape[0] - simm_mismatches_tr,
)
symm_err_test1 = M_test1_hat_ - np.transpose(M_test1_hat_, axes=(0, 2, 1))
simm_mismatches_test1 = np.sum(
    [
        np.allclose(M_test1_hat_[i, :, :], M_test1_hat_[i, :, :].T, rtol=0.0, atol=1e-5)
        for i in range(M_test1_hat_.shape[0])
    ]
)
print(
    "Number of test samples with non-symmetric inertia (tolerance 1e-5): ",
    M_test1_hat_.shape[0] - simm_mismatches_test1,
)

# check positivity
eig_tr, _ = np.linalg.eig(M_tr)
# eig_tr_hat, _  = np.linalg.eig(M_tr_hat)
eig_tr_hat_, _ = np.linalg.eig(M_tr_hat_)
eig_test1, _ = np.linalg.eig(M_test1)
# eig_test1_hat, _  = np.linalg.eig(M_test1_hat)
eig_test1_hat_, _ = np.linalg.eig(M_test1_hat_)
non_positive_def_count_tr = np.sum(np.min(eig_tr_hat_, 1) <= 0)
non_positive_def_count_test1 = np.sum(np.min(eig_test1_hat_, 1) <= 0)
print("Number of training samples with non-positive inertia: ", non_positive_def_count_tr)
print("Number of test samples with non-positive inertia: ", non_positive_def_count_test1)

# plot MSE
Project_Utils.plot_list([err_M_tr_, err_M_test1_], ["Training inertias MSE", "Test inertias MSE"], ["Training", "Test"])
# plt.show()

# plot eigenvals
Project_Utils.plot_eigenvals(
    [np.sort(eig_tr), np.sort(eig_tr_hat_)],
    colors=["k", "r"],
    labels=["True", "Estimated"],
    title="Training eigenvalues",
)
Project_Utils.plot_eigenvals(
    [np.sort(eig_test1), np.sort(eig_test1_hat_)],
    colors=["k", "r"],
    labels=["True", "Estimated"],
    title="Test eigenvalues",
)
# plt.show()


# plot diagonal elements of M
n_rows = int(np.ceil(num_dof / 2))
plt.figure()
plt.suptitle("Training inertias diag elements")
for i in range(num_dof):
    plt.subplot(n_rows, 2, i + 1)
    plt.ylabel("$M_{" + str(i + 1) + str(i + 1) + "}$")
    plt.plot(M_tr[:, i, i], label="true")
    plt.plot(M_tr_hat_[:, i, i], label="estimated")
    plt.grid()
    plt.legend()

plt.figure()
plt.suptitle("Test inertias diag elements")
for i in range(num_dof):
    plt.subplot(n_rows, 2, i + 1)
    plt.ylabel("$M_{" + str(i + 1) + str(i + 1) + "}$")
    plt.plot(M_test1[:, i, i], label="true")
    plt.plot(M_test1_hat_[:, i, i], label="estimated")
    plt.grid()
    plt.legend()

# plot determinant of principal minors of M
plt.figure()
plt.suptitle("Training inertias det of principal minors")
for i in range(num_dof):
    plt.subplot(n_rows, 2, i + 1)
    plt.ylabel("$det(M_{" + str(i + 1) + str(i + 1) + "})$")
    plt.plot(np.linalg.det(M_tr[:, : i + 1, : i + 1]), label="true")
    plt.plot(np.linalg.det(M_tr_hat_[:, : i + 1, : i + 1]), label="estimated")
    plt.grid()
    plt.legend()

plt.figure()
plt.suptitle("Test inertias det of principal minors")
for i in range(num_dof):
    plt.subplot(n_rows, 2, i + 1)
    plt.ylabel("$det(M_{" + str(i + 1) + str(i + 1) + "})$")
    plt.plot(np.linalg.det(M_test1[:, : i + 1, : i + 1]), label="true")
    plt.plot(np.linalg.det(M_test1_hat_[:, : i + 1, : i + 1]), label="estimated")
    plt.grid()
    plt.legend()

plt.show()

# GET ENERGY ESTIMATES #
print("\n\n----- GET ENERGY ESTIMATES -----")

# compute actual energy
Y_T_tr = 0.5 * np.array(
    [(input_tr[i, vel_indices].T.dot(M_tr[i, :, :])).dot(input_tr[i, vel_indices]) for i in range(input_tr.shape[0])]
).reshape(-1, 1)
Y_T_test1 = 0.5 * np.array(
    [
        (input_test1[i, vel_indices].T.dot(M_test1[i, :, :])).dot(input_test1[i, vel_indices])
        for i in range(input_test1.shape[0])
    ]
).reshape(-1, 1)

# get actual potential energy
if "U" in data_frame_tr.columns:
    U_tr = data_frame_tr["U"].to_numpy().reshape(-1, 1)
    U_test1 = data_frame_test1["U"].to_numpy().reshape(-1, 1)

# get energy estimates
with torch.no_grad():
    # Training
    print("\nEstimate the training energy...")
    Y_L_tr_hat, Y_T_tr_hat, Y_U_tr_hat, var_L_tr_hat, var_T_tr_hat, var_U_tr_hat = m.get_energy_estimate_from_alpha(
        X=input_tr[::downsampling], X_test=input_tr, alpha=alpha_tr, K_X_inv=None
    )
    print("\nEstimate the test energy...")
    (
        Y_L_test1_hat,
        Y_T_test1_hat,
        Y_U_test1_hat,
        var_L_test1_hat,
        var_T_test1_hat,
        var_U_test1_hat,
    ) = m.get_energy_estimate_from_alpha(X=input_tr[::downsampling], X_test=input_test1, alpha=alpha_tr, K_X_inv=None)
if "U" in data_frame_tr.columns:
    U_offset_tr = U_tr[0] - Y_U_tr_hat[0]
    U_offset_test1 = U_test1[0] - Y_U_test1_hat[0]


print("\nKinetic energy training ", end="")
Project_Utils.get_stat_estimate(Y=Y_T_tr, Y_hat=Y_T_tr_hat, stat_name="nMSE")
print("\nKinetic energy test ", end="")
Project_Utils.get_stat_estimate(Y=Y_T_test1, Y_hat=Y_T_test1_hat, stat_name="nMSE")

if "U" in data_frame_tr.columns:
    print("\nPotential energy training ", end="")
    Project_Utils.get_stat_estimate(Y=U_tr - U_offset_tr, Y_hat=Y_U_tr_hat, stat_name="nMSE")
    print("\nPotential energy test ", end="")
    Project_Utils.get_stat_estimate(Y=U_test1 - U_offset_test1, Y_hat=Y_U_test1_hat, stat_name="nMSE")

if "U" in data_frame_tr.columns:
    U_true_tr = U_tr - U_offset_tr
    L_true_tr = -U_tr + U_offset_tr + Y_T_tr
else:
    U_true_tr = None
    L_true_tr = None

if "U" in data_frame_tr.columns:
    U_true_test1 = U_test1 - U_offset_test1
    L_true_test1 = -U_test1 + U_offset_test1 + Y_T_test1
else:
    U_true_test1 = None
    L_true_test1 = None

Project_Utils.plot_energy(
    true_kin=Y_T_tr,
    true_pot=U_true_tr,
    true_lag=L_true_tr,
    est_kin=Y_T_tr_hat,
    est_pot=Y_U_tr_hat,
    est_lag=Y_L_tr_hat,
    title="Energy estimates on training trajectory",
)
Project_Utils.plot_energy(
    true_kin=Y_T_test1,
    true_pot=U_true_test1,
    true_lag=L_true_test1,
    est_kin=Y_T_test1_hat,
    est_pot=Y_U_test1_hat,
    est_lag=Y_L_test1_hat,
    title="Energy estimates on test trajectory",
)
plt.show()


if flg_frict and friction_model == "linear":
    print("\nFriction parameters estimation:")
    w_friction_list = m.get_friction_parameters(
        X_tr=torch.tensor(input_tr[::downsampling], dtype=dtype, device=device), alpha=alpha_tr
    )
    for joint_index in range(num_dof):
        print("Joint " + str(joint_index + 1) + " friction:", w_friction_list[joint_index])
