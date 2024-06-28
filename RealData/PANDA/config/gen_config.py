# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Script file for training/test file generation MC real
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)

"""

tr_seed_list = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 18]
test_seed_list = [21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
test_file_name = "panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0"
flg_gen_tr = True
flg_gen_test = True

tr_config_file_name = "Real_data_PANDA_config_training_comb"
test_config_file_name = "Real_data_PANDA_config_test_comb"
config_string_def = "[DEFAULT]\n\
# string values\n\
data_path = Experiments/\n\
file_name_1 = panda7dof_num_sin_50_seed_{}_as_filtered_fcut4.0.pkl,panda7dof_num_sin_50_seed_{}_as_filtered_fcut4.0.pkl\n\
file_name_2 = panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0.pkl,panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0.pkl\n\
loading_path = Results/50-100-sin_two-dataset_comb{}_with_linear_friction_\n\
saving_path = Results/50-100-sin_two-dataset_comb{}_with_linear_friction_\n\
flg_plot = False\n\
dev_name = cuda:0\n\
output_feature = tau_interp\n\
noiseless_output_feature = tau_interp\n\
robot_structure_str = rrrrrrr\n\
flg_frict = True\n\
downsampling = 1\n\
downsampling_data_load = 10\n\
batch_size = 250\n\
drop_last = True\n\
n_epoch = 751\n\
flg_np = True\n\
vel_threshold = -1\n\
# boolean values\n\
flg_load = {}\n\
flg_save = True\n\
flg_train = {}\n\
flg_norm = False\n\
flg_norm_noise = False\n\
flg_mean_norm = False\n\
single_sigma_n = False\n\
shuffle = True\n\
# int values\n\
num_dof = 7\n\
num_prism = 0\n\
num_rev = 7\n\
num_dat_tr = -1\n\
n_epoch_print = 50\n\
num_threads = 4\n\
#float values\n\
sigma_n_num = 0.0001\n\
lr = 0.01\n"

config_string_sec = "\n[LK_GIP_vel]\n\
lk_model_name = m_GP_LK_GIP\n\
f_k_name = get_K_blocks_GIP_vel_PANDA_7dof_no_subs\n\
[LK_GIP]\n\
LK_model_name = m_GP_LK_GIP\n\
f_k_name = get_K_blocks_GIP_PANDA_7dof_no_subs\n\
[LK_POLY_RBF]\n\
LK_model_name = m_GP_LK_POLY_RBF\n\
f_k_name = get_K_blocks_POLY_RBF7dof_no_subs\n\
[LK_POLY_vel_RBF]\n\
LK_model_name = m_GP_LK_POLY_RBF\n\
f_k_name = get_K_blocks_POLY_vel_RBF7dof_no_subs\n\
[LK_GIP_sum]\n\
lk_model_name = m_GP_LK_GIP_sum\n\
f_k_name = get_K_blocks_GIP_sum_PANDA_7dof_no_subs\n\
[LK_RBF_M]\n\
LK_model_name = m_GP_LK_RBF_M\n\
f_k_name = get_K_blocks_RBF_M_7dof_no_subs\n\
[LK_POLY_RBF_sum]\n\
LK_model_name = m_GP_LK_POLY_RBF_sum\n\
f_k_name = get_K_blocks_POLY_vel_RBF_sum7dof_no_subs\n\
[LK_RBF_1_sc]\n\
lk_model_name = m_GP_LK_RBF_1\n\
f_k_name = get_K_blocks_RBF_1_sc_7dof_no_subs"

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


# LIP config files
tr_seed_list = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 18]
test_seed_list = [21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
test_file_name = "panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0"
flg_gen_tr = True
flg_gen_test = True

tr_config_file_name = "Real_data_PANDA_LIP_config_training_comb"
test_config_file_name = "Real_data_PANDA_LIP_config_test_comb"
config_string_def = "[DEFAULT]\n\
# string values\n\
data_path = Experiments/\n\
file_name_1 = panda7dof_num_sin_50_seed_{}_as_filtered_fcut4.0.pkl,panda7dof_num_sin_50_seed_{}_as_filtered_fcut4.0.pkl\n\
file_name_2 = panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0.pkl,panda7dof_num_sin_100_seed_{}_as_filtered_fcut4.0.pkl\n\
loading_path = Results/50-100-sin_two-dataset_comb{}_with_NP_\n\
saving_path = Results/50-100-sin_two-dataset_comb{}_with_NP_\n\
flg_plot = False\n\
dev_name = cuda:0\n\
output_feature = tau_interp\n\
noiseless_output_feature = tau_interp\n\
robot_structure_str = rrrrrrr\n\
flg_frict = False\n\
downsampling = 1\n\
downsampling_data_load = 10\n\
batch_size = 250\n\
drop_last = True\n\
n_epoch = 751\n\
flg_np = True\n\
vel_threshold = -1\n\
# boolean values\n\
flg_load = {}\n\
flg_save = True\n\
flg_train = {}\n\
flg_norm = False\n\
flg_norm_noise = False\n\
flg_mean_norm = False\n\
single_sigma_n = False\n\
shuffle = True\n\
# int values\n\
num_dof = 7\n\
num_prism = 0\n\
num_rev = 7\n\
num_dat_tr = -1\n\
n_epoch_print = 50\n\
num_threads = 4\n\
#float values\n\
sigma_n_num = 0.0001\n\
lr = 0.01\n"

config_string_sec = "\n[LK_GIP_vel]\n\
lk_model_name = m_GP_LK_GIP\n\
f_k_name = get_K_blocks_GIP_vel_PANDA_7dof_no_subs\n\
[LK_GIP]\n\
LK_model_name = m_GP_LK_GIP\n\
f_k_name = get_K_blocks_GIP_PANDA_7dof_no_subs\n\
[LK_POLY_RBF]\n\
LK_model_name = m_GP_LK_POLY_RBF\n\
f_k_name = get_K_blocks_POLY_RBF7dof_no_subs\n\
[LK_POLY_vel_RBF]\n\
LK_model_name = m_GP_LK_POLY_RBF\n\
f_k_name = get_K_blocks_POLY_vel_RBF7dof_no_subs\n\
[LK_GIP_sum]\n\
lk_model_name = m_GP_LK_GIP_sum\n\
f_k_name = get_K_blocks_GIP_sum_PANDA_7dof_no_subs\n\
[LK_RBF_M]\n\
LK_model_name = m_GP_LK_RBF_M\n\
f_k_name = get_K_blocks_RBF_M_7dof_no_subs\n\
[LK_POLY_RBF_sum]\n\
LK_model_name = m_GP_LK_POLY_RBF_sum\n\
f_k_name = get_K_blocks_POLY_vel_RBF_sum7dof_no_subs\n\
[LK_RBF_1_sc]\n\
lk_model_name = m_GP_LK_RBF_1\n\
f_k_name = get_K_blocks_RBF_1_sc_7dof_no_subs"

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
