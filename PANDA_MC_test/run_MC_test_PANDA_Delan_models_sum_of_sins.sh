#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Giulio Giacomuzzo 20/08/2022
    # This script trains or test Delan models on the PANDA robot with increasing degrees of freedom (from 3 to 7)
    # @params:
        # - (string) [train/test/trai_test] : action to be performed

    # if train:
        # for n in {3..7}
            # train model on PANDA_$n_dof using trj with seed 0
            # and save it
    # if test:
        # for n in {3..7}
            # test the model on PANDA_$n_dof using trj with seed from 1 to 50
            # and save them
    # if train_test:
        # for n in {3..7}
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

        if [ $mode = 'train' ]
        then
            for seed in {1..50}
            do
                printf '\n################################################\n'
                date
                printf 'Training '$i$' dof Delan Model num_sins 50'
                    python DeLaN_estimator.py -r 0 -m 1 -l 0 -t 1 -c 0 -n ${i} -n_threads 12\
                    -saving_path ./Results/${i}dof/tr_num_sin50_train-test_num_sin50_seed${seed}_\
                    -model_loading_path ./Results/${i}dof/tr_num_sin50_train-test_num_sin50_seed${seed}_ \
                    -f1 ./data/${i}dof/panda_${i}dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed0.pkl \
                    -f2 ./data/${i}dof/panda_${i}dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed1.pkl \
                    -f1M ./data/${i}dof/panda_${i}dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed0.pkl \
                    -f2M ./data/${i}dof/panda_${i}dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed1.pkl
            done
        elif [ $mode = 'test' ]
        then
            for seed in {1..50}
            do

                printf '\n################################################\n'
                date
                printf '\nTesting '$i$' dof Delan Model seed '$((100+$seed))$' tr num_sins 50 test num_sins 50 \n'
                python DeLaN_estimator.py -r 0 -m 1 -l 1 -t 1 -c 0 -n ${i} \
                -saving_path ./Results/${i}dof/tr_num_sin50_train-test_num_sin50_seed${seed}_ \
                -model_loading_path ./Results/${i}dof/tr_num_sin50_train-test_num_sin50_seed${seed}_ \
                -f1 ./data/${i}dof/panda_${i}dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed45.pkl \
                -f2 ./data/${i}dof/panda_${i}dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed$(($seed + 100)).pkl \
                -f1M ./data/${i}dof/panda_${i}dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed49.pkl \
                -f2M ./data/${i}dof/panda_${i}dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed$(($seed + 100)).pkl

            done
        else
            for seed in {1..50}
            do
                printf '\n################################################\n'
                date
                printf '\nTraining '$i$' dof '$ker$' with 50 sins trj seed '$seed$', testing on 50 sins trj seed '$((100+$seed))$'\n'
                python DeLaN_estimator.py -r 0 -m 1 -l 0 -t 1 -c 0 -n ${i} \
                -saving_path ./Results/${i}dof/tr_num_sin50_train-test_num_sin50_seed${seed}_ \
                -model_loading_path ./Results/${i}dof/tr_num_sin50_train-test_num_sin50_seed${seed}_ \
                -f1 ./data/${i}dof/panda_${i}dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed${seed}.pkl \
                -f2 ./data/${i}dof/panda_${i}dof_data_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed$((100+$seed)).pkl \
                -f1M ./data/${i}dof/panda_${i}dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed${seed}.pkl \
                -f2M ./data/${i}dof/panda_${i}dof_data_M_inv_dyn_sum_of_sin__noise_std0.01__w_f0.02__num_sin50__pos_range_0.5__50seconds__Ts_0.01__seed$((100+$seed)).pkl

            done
        fi
    done

else
    echo 'Usage: MC_test_PANDA_Delan_models_sum_of_sins.sh [test/train/train_test]'
fi
