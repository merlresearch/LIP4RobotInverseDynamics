#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# before running make sure the config files have been generated
# $ cd ./config/
# $ python gen_config.py
# $ python gen_config_single_joint.py
# $ python gen_config_parametric_ID.py

# training and test of LK models
kernels=('LK_GIP_sum')
for ker in ${kernels[@]}
    do
        for i in 0 1 2 3 4 7 8 9 10
        do
            printf '\n################################################\n'
            date
            echo 'Training '$ker$' model combination '$i
            python ../../GP_estimator_real_data.py -config_file config/Real_data_new_MELFA_config_training_seed${i}.ini -kernel_name $ker
        done
    done

kernels=('LK_RBF_1_sc')
for ker in ${kernels[@]}
    do
        for i in 0 1 2 3 4 7 8 9 10
        do
            printf '\n################################################\n'
            date
            echo 'Training '$ker$' model combination '$i
            python ../../GP_estimator_real_data.py -config_file config/Real_data_new_MELFA_config_no_NP_training_seed${i}.ini -kernel_name $ker
        done
    done

# training and test of SJ models
kernels=('m_ind_RBF' 'm_ind_GIP_with_friction')
for ker in ${kernels[@]}
    do
        for i in 0 1 2 3 4 7 8 9 10
        do
            printf '\n################################################\n'
            date
            echo 'Training '$ker$' model combination '$i
            python ../../GP_estimator_single_joint.py -config_file config/Real_data_MELFA_single_joint_config_training_seed${i}.ini -kernel_name $ker
        done
    done

# parametric ID
for tr_seed in 0 1 2 3 4 7 8 9 10
do
    for test_seed in 1 2 3 4 6 7 8 9 11
    do
        python ../../Parametric_estimator.py -config_file 'config/Real_data_MELFA_parametric_ID_seed'${tr_seed}'_test_seed'${test_seed}.ini -robot_name 'MELFA'
    done
done
