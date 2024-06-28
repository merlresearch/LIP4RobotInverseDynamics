# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Script file for training/test file generation MC real
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)

"""

tr_seed_list = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 18]
test_seed_list = [21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
tr_config_file_name = "Real_data_PANDA_single_joint_config_training_comb"
test_config_file_name = "Real_data_PANDA_single_joint_config_test_comb"
test_file_name = "panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0"
flg_gen_tr = True
flg_gen_test = True

config_string_def = "[DEFAULT]\n\
# string values\n\
data_path = Experiments/\n\
file_name_1 = panda7dof_num_sin_50_seed_{}_as_filtered_fcut4.0.pkl,panda7dof_num_sin_50_seed_{}_as_filtered_fcut4.0.pkl\n\
file_name_2 = panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0.pkl,panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0.pkl\n\
loading_path = Results/50-100-sin_two-dataset_comb{}_\n\
saving_path = Results/50-100-sin_two-dataset_comb{}_\n\
dev_name = cuda:0\n\
output_feature = tau_interp\n\
noiseless_output_feature = tau_interp\n\
robot_structure_str = rrrrrrr\n\
vel_threshold = -1\n\
# boolean values\n\
flg_load = {}\n\
flg_save = True\n\
flg_train = {}\n\
flg_norm = False\n\
flg_norm_noise = False\n\
flg_mean_norm = False\n\
single_sigma_n = True\n\
shuffle = False\n\
flg_plot = False\n\
# int values\n\
num_dof = 7\n\
num_prism = 0\n\
num_rev = 7\n\
num_dat_tr = -1\n\
downsampling = 1\n\
downsampling_data_load = 10\n\
batch_size = -1\n\
n_epoch = 1501\n\
drop_last = False\n\
n_epoch_print = 250\n\
num_threads = 1\n\
#float values\n\
sigma_n_num = 0.0001\n\
lr = 0.01\n"

config_string_sec = "[m_ind_RBF]\n\
model_name = m_ind_RBF\n\
[m_ind_LIN]\n\
model_name = m_ind_LIN\n\
pp_function_module = simulated_envs.PANDA_sympybotics\n\
num_par = 72\n\
[m_ind_SP]\n\
model_name = m_ind_SP\n\
pp_function_module = simulated_envs.PANDA_sympybotics\n\
num_par = 72\n\
[m_ind_GIP]\n\
model_name = m_ind_GIP\n\
[m_ind_GIP_with_friction]\n\
model_name = m_ind_GIP_with_friction"

# generate training config files
if flg_gen_tr:
    for comb_index in range(len(tr_seed_list) - 1):
        f = open(tr_config_file_name + str(comb_index) + ".ini", "w")
        f.write(
            config_string_def.format(
                tr_seed_list[comb_index],
                tr_seed_list[comb_index + 1],
                test_seed_list[0],
                test_seed_list[1],
                comb_index,
                comb_index,
                "False",
                "True",
            )
            + config_string_sec
        )
        f.close()


# generate test config files
if flg_gen_test:
    test_file_list = [test_file_name.format(test_seed) for test_seed in test_seed_list]
    test_file_list_str = "\ntest_file_list =" + ",".join(test_file_list) + "\n"
    for comb_index in range(len(tr_seed_list) - 1):
        f = open(test_config_file_name + str(comb_index) + ".ini", "w")
        f.write(
            config_string_def.format(
                tr_seed_list[comb_index],
                tr_seed_list[comb_index + 1],
                test_seed_list[0],
                test_seed_list[1],
                comb_index,
                comb_index,
                "True",
                "False",
            )
            + test_file_list_str
            + config_string_sec
        )
        f.close()
