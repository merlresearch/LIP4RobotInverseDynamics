#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Giulio Giacomuzzo 20/08/2022
    # This script trains or test GP models with Lagrangian kernels on the PANDA robot with increasing degrees of freedom (from 3 to 7)
    # @params:
        # - (string) [train/test] : action to be performed

    # if train:
        # for n in {3..7}
            # for ker in {'LK_RBF_1_sc' 'LK_GIP_sum'}
                # train model on PANDA_$n_dof with $ker kernel using trj with seed 0
                # and save it
    # if test:
        # for n in {3..7}
            # for ker {'LK_RBF_1_sc' 'LK_GIP_sum'}
                # test the model on PANDA_$n_dof with $ker kernel using trj with seed from 1 to 50
                # and save them

    # if train-test:
        # for n in {3..7}
            # for ker in {'LK_RBF_1_sc' 'LK_GIP_sum'}
                # for seed in {1..50}
                # train the model on PANDA_$n_dof with $ker kernel using $seed and test it on ${100+seed}
                # and save them

if [ $# -eq 1 ]  && [[ $1 = 'train' || $1 = 'test' || $1 = 'train_test' ]]
then
    mode=$1

    for i in {3..7}
    # for i in 6
    do
        if [ ! -d './Results/'$i$'dof' ]
        then
            echo "Directory 'Results/'$i$'dof' does not exists. Creating it."
            mkdir -p './Results/'$i$'dof'
        fi


        # kernels=('LK_POLY_RBF' 'LK_GIP_vel' 'LK_GIP_sum')
        # kernels=('LK_GIP_sum')
        kernels=('LK_RBF_1_sc' 'LK_GIP_sum')


        if [ $mode = 'train' ]
        then
            for ker in ${kernels[@]}
            do
                printf '\n################################################\n'
                date
                echo 'Training '$i$' dof '$ker$' num sin 50'
                python LK_GP_estimator_MC.py -config_file './config/PANDA_'$i$'dof_MC_config_tr_num_sin50.ini' -kernel_name $ker

            done
        elif   [ $mode = 'test' ]
        then
            for seed in {101..150}
            do
                for ker in ${kernels[@]}
                do
                    printf '\n################################################\n'
                    date
                    printf '\nTesting '$i$' dof '$ker$' seed '$seed$' tr 50 sins test 50 sins\n'
                    python LK_GP_estimator_MC.py -config_file './config/PANDA_'$i$'dof_MC_config_test_tr_num_sin50_test_num_sin50__seed'$seed'.ini' -kernel_name $ker

                done
            done
        else
            for seed in {1..50}
            do
                for ker in ${kernels[@]}
                do
                    printf '\n################################################\n'
                    date
                    printf '\nTraining '$i$' dof '$ker$' with 50 sins trj seed '$seed$', testing on 50 sins trj seed '$((100+$seed))$'\n'
                    python LK_GP_estimator_MC.py -config_file './config/PANDA_'$i$'dof_MC_config_train-test_tr_num_sin50_test_num_sin50__seed'$seed'.ini' -kernel_name $ker
                done
            done
        fi
    done

else
    echo 'Usage: MC_test_PANDA_LK_models.sh [test/train/train_test]'
fi
