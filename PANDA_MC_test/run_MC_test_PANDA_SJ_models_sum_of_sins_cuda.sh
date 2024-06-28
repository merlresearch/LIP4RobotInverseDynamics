#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Giulio Giacomuzzo 20/08/2022
    # This script trains or test single joint GP models the simulated PANDA robot with increasing degrees of freedom (from 3 to 7)
    # @params:
        # - (string) [train/test/train_test] : action to be performed

    # if train:
        # for n in {3..7}
            # for ker in {'RBF' 'GIP'}
                # train model on PANDA_$n_dof with $ker kernel using trj with seed 0
                # and save it
    # if test:
        # for n in {3..7}
            # for ker {'RBF' 'GIP'}
                # test the model on PANDA_$n_dof with $ker kernel using trj with seed from 1 to 50
                # and save them

    # if train_test:
        # for n in {3..7}
            # for ker in {'RBF' 'GIP'}
                # for seed in {1..50}
                # train the model on PANDA_$n_dof with $ker kernel using $seed and test it on ${100+seed}
                # and save them

if [ $# -eq 1 ]  && [[ $1 = 'train' || $1 = 'test' || $1 = 'train_test' ]]
then
    mode=$1

    for i in {3..7}
    do
        if [ ! -d './Results/'$i$'dof' ]
        then
            echo "Directory 'Results/'$i$'dof' does not exists. Creating it."
            mkdir -p './Results/'$i$'dof'
        fi


        kernels=('m_ind_GIP' 'm_ind_RBF')


        if [ $mode = 'train' ]
        then
            for ker in ${kernels[@]}
            do
                printf '\n################################################\n'
                date
                echo 'Training '$i$' dof '$ker$' num sin 50'
                python GP_estimator_single_joint.py -config_file './config/PANDA_'$i$'dof_MC_config_tr_num_sin50__SJ__cuda.ini' -kernel_name $ker

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
                    python GP_estimator_single_joint.py -config_file './config/PANDA_'$i$'dof_MC_config_test_tr_num_sin50_test_num_sin50__seed'$seed'__SJ__cuda.ini' -kernel_name $ker

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
                    python GP_estimator_single_joint.py -config_file './config/PANDA_'$i$'dof_MC_config_train-test_tr_num_sin50_test_num_sin50__seed'$seed'__SJ__cuda.ini' -kernel_name $ker
                done
            done
        fi
    done

else
    echo 'Usage: ./run_MC_test_PANDA_SJ_models_sum_of_sins_cuda [test/train/train_test]'
fi
