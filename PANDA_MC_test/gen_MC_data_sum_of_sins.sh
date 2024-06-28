#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Giulio Giacomuzzo 20/08/2022
    # This script generates trajectories of the PANDA robot with increasing dregrees of freedom (from 3 to 7)
    # Trajectories contains  joint position, velocities, accelerations and torques,
    # inertia matrices and potential energy (kinetic energy is easily computed from velocities and inertia matrices)
    # seed 0 is intended for training, other seeds for test.
    # By default the number of testing seed is 50 [0..50], but it can be passed as an argument
    # @params:
        # - (int) num_seeds : number of test trajectories to be generated

if [ $# = 1 ]
then
    num_seed=$1
else
    num_seed=10
fi

if [ ! -d 'data/' ]
then
    echo "Directory 'data/' does not exists. Creating it."
    mkdir -p './data/'
fi
for i in {3..7}
do
    echo 'Generating MC data: '$i$'dof'
    if [ ! -d './data/'$i$'dof' ]
    then
        echo " Directory 'data/'$i$'dof' does not exists. Creating it."
        mkdir -p './data/'$i$'dof'
    fi
    saving_path='./data/'$i$'dof/'
    for (( seed=0; seed<=$num_seed; seed++ ))
    do
        echo ' seed: '$seed$' 50 sinusoids'
        python ../simulated_envs/PANDA_sympybotics/Get_PANDA_data_inv_dyn_sum_of_sin.py -saving_path $saving_path -pos_range 0.5 -c_init 0.1 -num_sin 50 -seed $seed -std_noise 0.01 -sampling_time 0.01 -num_active_joints $i -plot 0

        # echo ' seed: '$seed$' 100 sinusoids'
        # python ../simulated_envs/PANDA_sympybotics/Get_PANDA_data_inv_dyn_sum_of_sin.py -saving_path $saving_path -pos_range 0.8 -c_init 0.1 -num_sin 100 -seed $seed -std_noise 0.01 -sampling_time 0.01 -num_active_joints $i -plot 0
    done

    for (( seed=100; seed<=100+$num_seed; seed++ ))
    do
        echo ' seed: '$seed$' 50 sinusoids'
        python ../simulated_envs/PANDA_sympybotics/Get_PANDA_data_inv_dyn_sum_of_sin.py -saving_path $saving_path -pos_range 0.5 -c_init 0.1 -num_sin 50 -seed $seed -std_noise 0.01 -sampling_time 0.01 -num_active_joints $i -plot 0

        # echo ' seed: '$seed$' 100 sinusoids'
        # python ../simulated_envs/PANDA_sympybotics/Get_PANDA_data_inv_dyn_sum_of_sin.py -saving_path $saving_path -pos_range 0.8 -c_init 0.1 -num_sin 100 -seed $seed -std_noise 0.01 -sampling_time 0.01 -num_active_joints $i -plot 0
    done

done
