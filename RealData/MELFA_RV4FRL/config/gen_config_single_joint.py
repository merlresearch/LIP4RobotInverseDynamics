# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Script file for training/test file generation MC real
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)

"""

seed_list = [0, 1, 2, 3, 4, 7, 8, 9, 10]
test_seed_list = [1, 2, 3, 4, 6, 7, 8, 9, 11]
tr_config_file_name = "Real_data_MELFA_single_joint_config_training_seed"
test_config_file_name = "Real_data_MELFA_single_joint_config_test_seed"
test_file_name = "SumOfSin__w_f0.02__num_sin100__pos_range_1.4__50seconds__seed"
flg_gen_tr = True
flg_gen_test = True

config_string_def = "[DEFAULT]\n\
# string values\n\
data_path = Experiments/\n\
file_name_1 = SumOfSin__w_f0.02__num_sin100__pos_range_0.5__50seconds__seed{}.pkl\n\
file_name_2 = SumOfSin__w_f0.02__num_sin100__pos_range_0.5__50seconds__seed{}.pkl\n\
loading_path =  Results/100SmallRange-100LargeRange_seed{}_low_vel_\n\
saving_path =  Results/100SmallRange-100LargeRange_seed{}_low_vel_\n\
dev_name = cuda:0\n\
output_feature = tau\n\
noiseless_output_feature = tau\n\
robot_structure_str = rrrrrr\n\
vel_threshold = 0.01\n\
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
num_dof = 6\n\
num_prism = 0\n\
num_rev = 6\n\
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
[m_ind_GIP]\n\
model_name = m_ind_GIP\n\
[m_ind_GIP_with_friction]\n\
model_name = m_ind_GIP_with_friction"

test_file_list = [test_file_name + str(test_seed) for test_seed in test_seed_list]
test_file_list_str = "\ntest_file_list = " + ",".join(test_file_list) + "\n"

# generate training config files
if flg_gen_tr:
    for seed in seed_list:
        f = open(tr_config_file_name + str(seed) + ".ini", "w")
        f.write(
            config_string_def.format(seed, seed, seed, seed, "False", "True") + test_file_list_str + config_string_sec
        )
        f.close()


# generate test config files
if flg_gen_test:
    for seed in seed_list:
        f = open(test_config_file_name + str(seed) + ".ini", "w")
        f.write(
            config_string_def.format(seed, seed, seed, seed, "True", "False") + test_file_list_str + config_string_sec
        )
        f.close()
