#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Giulio Giacomuzzo 08/12/2022
    # This script runs the whole MC test (with gpu computations)
    # 1 - Generate synthetic data
    # 2 - generate config files
    # 3 - train & test LK models (LK_POLY_RBF_sum, LK_GIP_sum)
    # 4 - train & test SJ models (RBF, GIP)
    # 5 - train & test Delan model


if [ ! -d './log' ]
        then
            echo "Directory 'log' does not exists. Creating it."
            mkdir -p './log'
        fi

source gen_num_kernel_functions.sh | tee -a ./log/MC_test_cuda_log.txt
source gen_MC_data_sum_of_sins.sh 50 | tee -a ./log/MC_test_cuda_log.txt
python ./config/gen_config_file_sum_of_sin_cuda.py -name_prefix 'config/' | tee -a ./log/MC_test_cuda_log.txt
python ./config/gen_config_file_sum_of_sin_SJ_cuda.py -name_prefix 'config/' | tee -a ./log/MC_test_cuda_log.txt
source run_MC_test_PANDA_LK_models_sum_of_sins_cuda.sh test | tee -a ./log/MC_test_cuda_log.txt
source run_MC_test_PANDA_SJ_models_sum_of_sins_cuda.sh test | tee -a ./log/MC_test_cuda_log.txt
source run_MC_test_PANDA_Delan_models_sum_of_sins_cuda.sh test | tee -a ./log/MC_test_cuda_log.txt
