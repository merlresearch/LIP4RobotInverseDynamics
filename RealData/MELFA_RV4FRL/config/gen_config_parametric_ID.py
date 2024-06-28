# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Script file for training/test file generation MC real
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)

"""

seed_list = [0, 1, 2, 3, 4, 7, 8, 9, 10]
test_seed_list = [1, 2, 3, 4, 6, 7, 8, 9, 11]
config_file_name = "Real_data_MELFA_parametric_ID_seed"
test_file_name = "SumOfSin__w_f0.02__num_sin100__pos_range_1.4__50seconds__seed"

config_string = "[DEFAULT]\n\
# string values\n\
data_path = Experiments/\n\
file_name_1 = SumOfSin__w_f0.02__num_sin100__pos_range_0.5__50seconds__seed{}.pkl\n\
file_name_2 = {}.pkl\n\
loading_path =  Results/100SmallRange-100LargeRange_seed{}_low_vel_\n\
saving_path =  Results/100SmallRange-100LargeRange_seed{}_low_vel_\n\
output_feature = tau\n\
noiseless_output_feature = tau\n\
vel_threshold = 0.01\n\
# boolean values\n\
flg_load = False\n\
flg_save = True\n\
flg_norm = False\n\
flg_norm_noise = False\n\
flg_mean_norm = False\n\
flg_plot = False\n\
# int values\n\
num_dof = 6\n\
num_dat_tr = -1\n\
downsampling = 1\n\
downsampling_data_load = 10\n\
\n[MELFA]\n\
num_par = 48\n\
function_module = RealData.MELFA_RV4FRL"

# generate config files
for seed in seed_list:
    for seed_test in test_seed_list:
        test_file = test_file_name + str(seed_test)
        f = open(config_file_name + str(seed) + "_test_seed" + str(seed_test) + ".ini", "w")
        f.write(config_string.format(seed, test_file, str(seed) + "_" + test_file, str(seed) + "_" + test_file))
        f.close()
