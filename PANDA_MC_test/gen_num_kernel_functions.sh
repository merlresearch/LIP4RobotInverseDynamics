#!/bin/bash
# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
robot_structure="00"
for i in {3..7}
do
    robot_structure=$robot_structure"0"
    echo
    echo "-------------------------------------------------------------------------------------------------------------"
    date
    printf "Generating PANDA "$i" dof LIP numeric functions\n"
    python ../gpr_lib/GP_prior/LK/Compute_lagrangian_kernel_LIP.py -robot_name "PANDA_"$i"dof" \
                -robot_structure $robot_structure\
                -path "../gpr_lib/GP_prior/LK/"

    printf "\nGenerating PANDA "$i" dof LSE numeric functions\n"
    python ../gpr_lib/GP_prior/LK/Compute_lagrangian_kernel_LSE.py -robot_name "PANDA_"$i"dof" \
                -robot_structure $robot_structure\
                -path "../gpr_lib/GP_prior/LK/"

    echo "-------------------------------------------------------------------------------------------------------------"
done
