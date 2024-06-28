#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# before running make sure the config files have been generated
# $ cd ./config/PANDA/MC_TEST_REAL/
# $ python gen_config.py
# $ python gen_config_single_joint.py

# training and test of LK models
    for i in {0..11}
    do
        printf '\n################################################\n'
        date
        echo 'Training '$ker$' model combination '$i
        python ../../GP_estimator_real_data.py -config_file ./config/Real_data_PANDA_LIP_config_training_comb${i}.ini -kernel_name 'LK_GIP_sum'

        printf '\n################################################\n'
        date
        echo 'Testing '$ker$' model combination '$i
        python ../../GP_estimator_real_data.py -config_file ./config/Real_data_PANDA_LIP_config_test_comb${i}.ini -kernel_name 'LK_GIP_sum'
    done

    for i in {0..11}
    do
        printf '\n################################################\n'
        date
        echo 'Training '$ker$' model combination '$i
        python ../../GP_estimator_real_data.py -config_file ./config/Real_data_PANDA_config_training_comb${i}.ini -kernel_name 'LK_RBF_1_sc'

        printf '\n################################################\n'
        date
        echo 'Testing '$ker$' model combination '$i
        python ../../GP_estimator_real_data.py -config_file ./config/Real_data_PANDA_LIP_config_test_comb${i}.ini -kernel_name 'LK_RBF_1_sc'
    done

# training and test of SJ models
kernels=('m_ind_RBF' 'm_ind_GIP_with_friction')
for ker in ${kernels[@]}
    do
        for i in {0..11}
        do
            printf '\n################################################\n'
            date
            echo 'Training '$ker$' model combination '$i
            python ../../GP_estimator_single_joint.py -config_file ./config/Real_data_PANDA_single_joint_config_training_comb${i}.ini -kernel_name $ker

            printf '\n################################################\n'
            date
            echo 'Testing '$ker$' model combination '$i
            python ../../GP_estimator_single_joint.py -config_file ./config/Real_data_PANDA_single_joint_config_test_comb${i}.ini -kernel_name $ker
        done
    done
