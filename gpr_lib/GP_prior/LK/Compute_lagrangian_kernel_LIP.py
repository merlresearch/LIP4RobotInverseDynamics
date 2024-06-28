"""
Author:
Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
Giulio Giacomuzzo (giulio.giacomuzzo@gmail.com)
LIP kernel function generator
"""

import argparse

import Get_Lagrangian_kernel

# read argparse parameters
p = argparse.ArgumentParser("LIP kernel functions generation")
robot_structure = "0000000"
robot_name = "PANDA_7dof"
path = ""
p.add_argument(
    "-robot_structure",
    type=str,
    default="0000000",
    help="robot structure: one digit for each dof,\
               0 if revolute, 1 if prismatic",
)
p.add_argument("-robot_name", type=str, default="PANDA", help="robot name. Used to generate compiled functions")
p.add_argument("-path", type=str, default="", help="relative path where to save generated functions")
d_argparse = vars(p.parse_known_args()[0])
locals().update(vars(p.parse_known_args()[0]))

# ------ PANDA robot ------
n = len(robot_structure)

flg_subs = False
f_name = "get_K_blocks_GIP_sum_" + robot_name + "_no_subs" if not flg_subs else "get_K_blocks_GIP_sum_" + robot_name

optimizations = "basic"  # None or 'basic'

f_K_L, f_K_L_L, args, f_K_L_Y, f_K_T_Y, f_K_U_Y = Get_Lagrangian_kernel.get_Lagrangian_kernel_GIP_sum(
    n, robot_structure, flg_subs
)
Get_Lagrangian_kernel.gen_num_function_GIP_sum(
    f_K_L_L=f_K_L_L,
    f_name=f_name,
    n=n,
    robot_structure=robot_structure,
    optimizations=optimizations,
    flg_subs=flg_subs,
    f_K_L_Y=f_K_L_Y,
    f_K_T_Y=f_K_T_Y,
    f_K_U_Y=f_K_U_Y,
    path=path,
)
