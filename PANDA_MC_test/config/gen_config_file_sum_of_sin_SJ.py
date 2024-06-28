# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Script file that generates config files for Monte Carlo test on PANDA robot with increasing degrees of freedom
Author: Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)

"""

import argparse

name_prefix = ""
parser = argparse.ArgumentParser()
parser.add_argument("-name_prefix", type=str, required=False, default="", help="prefix for file name")
locals().update(vars(parser.parse_known_args()[0]))

for n in range(3, 8):
    for num_sin in [50, 100]:
        tr_config_file_name = name_prefix + "PANDA_" + str(n) + "dof_MC_config_tr_num_sin" + str(num_sin) + "__SJ.ini"

        tr_config_file_str = (
            """[DEFAULT]
# string values
data_path = ./data/"""
            + str(n)
            + """dof/
file_name_1 = panda_"""
            + str(n)
            + """dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
            + str(num_sin)
            + """__pos_range_0.5__50seconds__Ts_0.01__seed0.pkl
file_name_m1 = panda_"""
            + str(n)
            + """dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
            + str(num_sin)
            + """__pos_range_0.5__50seconds__Ts_0.01__seed0.pkl
file_name_2 = panda_"""
            + str(n)
            + """dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
            + str(num_sin)
            + """__pos_range_0.5__50seconds__Ts_0.01__seed10.pkl
file_name_m2 = panda_"""
            + str(n)
            + """dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
            + str(num_sin)
            + """__pos_range_0.5__50seconds__Ts_0.01__seed10.pkl
model_loading_path = ./Results/"""
            + str(n)
            + """dof/
saving_path = ./Results/"""
            + str(n)
            + """dof/num_sin_"""
            + str(num_sin)
            + """_
dev_name = cpu
output_feature = tau
noiseless_output_feature = tau_noiseless
robot_structure_str = rrrrrrr

# boolean values
flg_load = False
flg_save = True
flg_save_trj = False
flg_train = True
flg_norm = False
flg_norm_noise = False
flg_mean_norm = False
single_sigma_n = True
shuffle = True
drop_last=True
flg_plot=False

# int values
num_dof = """
            + str(n)
            + """
num_prism = 0
num_rev = """
            + str(n)
            + """
num_dat_tr = -1
downsampling = 1
downsampling_data_load = 10
batch_size = 100
n_epoch = 1501
n_epoch_print = 250
num_threads = 4

#float values
sigma_n_num = 0.0002
lr = 0.01
vel_threshold = -1

[m_ind_RBF]
model_name = m_ind_RBF

[m_ind_GIP]
model_name = m_ind_GIP

"""
        )

        with open(tr_config_file_name, "w") as f:
            f.write(tr_config_file_str)

    for seed in range(51):
        for ranges in [(50, 50), (100, 50), (100, 100)]:
            test_config_file_name = (
                name_prefix
                + "PANDA_"
                + str(n)
                + "dof_MC_config_test_tr_num_sin"
                + str(ranges[0])
                + "_test_num_sin"
                + str(ranges[1])
                + "__seed"
                + str(100 + seed)
                + "__SJ.ini"
            )
            test_config_file_str = (
                """[DEFAULT]
# string values
data_path = ./data/"""
                + str(n)
                + """dof/
file_name_1 = panda_"""
                + str(n)
                + """dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[0])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(seed)
                + """.pkl
file_name_m1 = panda_"""
                + str(n)
                + """dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[0])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(seed)
                + """.pkl
file_name_2 = panda_"""
                + str(n)
                + """dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[1])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(100 + seed)
                + """.pkl
file_name_m2 = panda_"""
                + str(n)
                + """dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[1])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(100 + seed)
                + """.pkl
model_loading_path = ./Results/"""
                + str(n)
                + """dof/tr_num_sin"""
                + str(ranges[0])
                + """_train-test_num_sin"""
                + str(ranges[1])
                + """_seed"""
                + str(seed)
                + """_
saving_path = ./Results/"""
                + str(n)
                + """dof/tr_num_sin"""
                + str(ranges[0])
                + """_train-test_num_sin"""
                + str(ranges[1])
                + """_seed"""
                + str(seed)
                + """_
dev_name = cpu
output_feature = tau
noiseless_output_feature = tau_noiseless
robot_structure_str = rrrrrrr

# boolean values
flg_load = True
flg_save = True
flg_save_trj = True
flg_train = False
flg_norm = False
flg_norm_noise = False
flg_mean_norm = False
single_sigma_n = True
shuffle = True
drop_last=True
flg_plot=False

# int values
num_dof = """
                + str(n)
                + """
num_prism = 0
num_rev = """
                + str(n)
                + """
num_dat_tr = -1
downsampling = 1
downsampling_data_load = 10
batch_size = 100
n_epoch = 1501
n_epoch_print = 250
num_threads = 4

#float values
sigma_n_num = 0.0002
lr = 0.01
vel_threshold = -1

[m_ind_RBF]
model_name = m_ind_RBF

[m_ind_GIP]
model_name = m_ind_GIP
"""
            )
            with open(test_config_file_name, "w") as f:
                f.write(test_config_file_str)

    for seed in range(51):
        # for ranges in [(50, 50), (100, 50), (100, 100)]:
        for ranges in [(50, 50)]:

            test_config_file_name = (
                name_prefix
                + "PANDA_"
                + str(n)
                + "dof_MC_config_train-test_tr_num_sin"
                + str(ranges[0])
                + "_test_num_sin"
                + str(ranges[1])
                + "__seed"
                + str(seed)
                + "__SJ.ini"
            )
            test_config_file_str = (
                """[DEFAULT]
# string values
data_path = ./data/"""
                + str(n)
                + """dof/
file_name_1 = panda_"""
                + str(n)
                + """dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[0])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(seed)
                + """.pkl
file_name_m1 = panda_"""
                + str(n)
                + """dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[0])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(seed)
                + """.pkl
file_name_2 = panda_"""
                + str(n)
                + """dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[1])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(100 + seed)
                + """.pkl
file_name_m2 = panda_"""
                + str(n)
                + """dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin"""
                + str(ranges[1])
                + """__pos_range_0.5__50seconds__Ts_0.01__seed"""
                + str(100 + seed)
                + """.pkl
model_loading_path = ./Results/"""
                + str(n)
                + """dof/num_sin_"""
                + str(ranges[0])
                + """_
saving_path = ./Results/"""
                + str(n)
                + """dof/tr_num_sin"""
                + str(ranges[0])
                + """_train-test_num_sin"""
                + str(ranges[1])
                + """_seed"""
                + str(seed)
                + """_
dev_name = cpu
output_feature = tau
noiseless_output_feature = tau_noiseless
robot_structure_str = rrrrrrr

# boolean values
flg_load = False
flg_save = True
flg_save_trj = True
flg_train = True
flg_norm = False
flg_norm_noise = False
flg_mean_norm = False
single_sigma_n = True
shuffle = True
drop_last=True
flg_plot=False

# int values
num_dof = """
                + str(n)
                + """
num_prism = 0
num_rev = """
                + str(n)
                + """
num_dat_tr = -1
downsampling = 1
downsampling_data_load = 10
batch_size = 100
n_epoch = 1501
n_epoch_print = 250
num_threads = 4

#float values
sigma_n_num = 0.0002
lr = 0.01
vel_threshold = -1


[m_ind_RBF]
model_name = m_ind_RBF

[m_ind_GIP]
model_name = m_ind_GIP
"""
            )
            with open(test_config_file_name, "w") as f:
                f.write(test_config_file_str)
