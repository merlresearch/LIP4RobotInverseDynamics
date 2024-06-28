# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "../../")
import Project_Utils_ as Project_Utils


def remove_exponential_zero(num):
    val = num[:-2]
    last = num[-1]
    return val + last


np.set_printoptions(suppress=True, precision=2, linewidth=500, formatter={"float_kind": lambda x: "{:.2f}".format(x)})

data_folder = "../Results/"
model_names = ["m_ind_RBF", "m_ind_GIP", "LK_RBF_1_sc", "LK_GIP_sum", "Delan"]
num_sin = [(50, 50)]
seeds = range(1, 51)
score = "nMSE"
curr_dict = {}
T_nmse_dict_6dof = {}
V_nmse_dict_6dof = {}
for n in range(3, 8):

    exec("nmse_dict_" + str(n) + "dof = dict()")
    exec("T_nmse_dict_" + str(n) + "dof = dict()")
    exec("V_nmse_dict_" + str(n) + "dof = dict()")
    exec("cum_nmse_dict_" + str(n) + "dof = dict()")

    for model in model_names:
        rng_nmse_list = []
        T_num_sin_list = []
        V_num_sin_list = []
        gmse_list = []
        for rng in num_sin:
            seed_T_nmse_list = []
            seed_V_nmse_list = []
            seed_nmse_list = []
            seed_gmse_list = []
            for seed in seeds:
                print("Extracting data: Panda {} dof, model {}, num sins {}, seed {}".format(n, model, rng, seed))
                # if "Delan" in model:
                #     if n == 4 or n == 5:
                #         file_name = (
                #             data_folder
                #             + str(n)
                #             + "dof/"
                #             + "tr_num_sin"
                #             + str(rng[0])
                #             + "_test3_num_sin"
                #             + str(rng[1])
                #             + "_seed"
                #             + str(seed)
                #             + "_model_"
                #             + model
                #             + "_test1_estimates.pkl"
                #         )
                #     else:
                #         file_name = (
                #             data_folder
                #             + str(n)
                #             + "dof/"
                #             + "tr_num_sin"
                #             + str(rng[0])
                #             + "_test2_num_sin"
                #             + str(rng[1])
                #             + "_seed"
                #             + str(seed)
                #             + "_model_"
                #             + model
                #             + "_test1_estimates.pkl"
                #         )
                # else:
                #     # if "m_ind" in model:
                #     #     file_name = (
                #     #         data_folder
                #     #         + str(n)
                #     #         + "dof/"
                #     #         + "tr_num_sin"
                #     #         + str(rng[0])
                #     #         + "_train-test_num_sin"
                #     #         + str(rng[1])
                #     #         + "_seed"
                #     #         + str(seed)
                #     #         + "_model_"
                #     #         + model
                #     #         + "_test1_estimates.pkl"
                #     #     )
                #     # else:
                file_name = (
                    data_folder
                    + str(n)
                    + "dof/"
                    + "tr_num_sin"
                    + str(rng[0])
                    + "_train-test_num_sin"
                    + str(rng[1])
                    + "_seed"
                    + str(seed)
                    + "_model_"
                    + model
                    + "_test1_estimates.pkl"
                )

                data = pkl.load(open(file_name, "rb"))

                if "m_ind" in model:
                    data_dict = data[1]
                else:
                    data_dict = data[0]
                nmse = Project_Utils.get_stat_estimate(
                    Y=data_dict["Y_noiseless"], Y_hat=data_dict["Y_hat"], stat_name=score, flg_print=False
                )

                if "T" in data_dict.keys():
                    T = data_dict["T"]
                    U = data_dict["U"]
                    T_nmse = Project_Utils.get_stat_estimate(
                        Y=T, Y_hat=data_dict["T_hat"].reshape(-1, 1), stat_name=score, flg_print=False
                    )
                    V_nmse = Project_Utils.get_stat_estimate(
                        Y=U, Y_hat=data_dict["U_hat"].reshape(-1, 1), stat_name=score, flg_print=False
                    )
                    seed_T_nmse_list.append(T_nmse)
                    seed_V_nmse_list.append(V_nmse)
                    # print(T_nmse)ip
                seed_nmse_list.append(nmse)
                seed_gmse_list.append(np.mean(nmse))
                # cum_nmse_list.append(np.sum(nmse))

            nmse_array = np.array(seed_nmse_list)
            gmse_array = np.array(seed_gmse_list)
            if "T" in data[0].keys():
                T_nmse_array = np.array(seed_T_nmse_list)
                V_nmse_array = np.array(seed_V_nmse_list)
                T_num_sin_list.append(T_nmse_array)
                V_num_sin_list.append(V_nmse_array)
            # cum_nmse_array = np.array(cum_nmse_list)
            rng_nmse_list.append(nmse_array)
            gmse_list.append(gmse_array)
        exec("nmse_dict_" + str(n) + "dof['" + model + "'] = rng_nmse_list")
        exec("T_nmse_dict_" + str(n) + "dof['" + model + "'] = T_num_sin_list")
        exec("V_nmse_dict_" + str(n) + "dof['" + model + "'] = V_num_sin_list")
        exec("cum_nmse_dict_" + str(n) + "dof['" + model + "'] = gmse_list")


colors = ["red", "yellow", "green", "blue", "purple"]
fig_cum, axes_cum = plt.subplots(1, 5, figsize=(10, 2))
for ndof in range(3, 8):
    exec("curr_dict = nmse_dict_" + str(ndof) + "dof")
    exec("curr_dict_g = cum_nmse_dict_" + str(ndof) + "dof")
    print("##########################################")
    print(str(ndof) + "dof")
    print(
        "SJ RBF GMSE: " + "{:.2f}".format(np.median(np.mean(curr_dict["m_ind_RBF"][0], axis=1))),
        np.percentile(np.mean(curr_dict["m_ind_RBF"][0], axis=1), [25, 75]),
    )
    print(
        "SJ GIP GMSE: " + "{:.2f}".format(np.median(np.mean(curr_dict["m_ind_GIP"][0], axis=1))),
        np.percentile(np.mean(curr_dict["m_ind_GIP"][0], axis=1), [25, 55]),
    )
    print(
        "LK RBF GMSE: " + "{:.2f}".format(np.median(np.mean(curr_dict["LK_RBF_1_sc"][0], axis=1))),
        np.percentile(np.mean(curr_dict["LK_RBF_1_sc"][0], axis=1), [25, 75]),
    )
    print(
        "LK GIP GMSE: " + "{:.2f}".format(np.median(np.mean(curr_dict["LK_GIP_sum"][0], axis=1))),
        np.percentile(np.mean(curr_dict["LK_GIP_sum"][0], axis=1), [25, 75]),
    )
    print(
        "Delan GMSE: " + "{:.2f}".format(np.median(np.mean(curr_dict["Delan"][0], axis=1))),
        np.percentile(np.mean(curr_dict["Delan"][0], axis=1), [25, 75]),
    )

    fig, axes = Project_Utils.generate_subplots(
        ndof, sharex=False, sharey="row", row_wise=True, figsize=(7, 4.5), horizontal=True
    )

    ncols_fig = int(np.ceil(ndof / 2))
    for dof, ax in zip(range(ndof), axes):

        # print stat
        if ndof == 6:
            print("")
            print("Joint " + str(dof + 1) + "stats:")
            print(
                "SJ RBF nMSE: " + "{:.2f}".format(np.median(curr_dict["m_ind_RBF"][0][:, dof])),
                np.percentile(curr_dict["m_ind_RBF"][0][:, dof], [25, 75]),
            )
            print(
                "SJ GIP nMSE: " + "{:.2f}".format(np.median(curr_dict["m_ind_GIP"][0][:, dof])),
                np.percentile(curr_dict["m_ind_GIP"][0][:, dof], [25, 55]),
            )
            print(
                "LK RBF nMSE: " + "{:.2f}".format(np.median(curr_dict["LK_RBF_1_sc"][0][:, dof])),
                np.percentile(curr_dict["LK_RBF_1_sc"][0][:, dof], [25, 75]),
            )
            print(
                "LK GIP nMSE: " + "{:.2f}".format(np.median(curr_dict["LK_GIP_sum"][0][:, dof])),
                np.percentile(curr_dict["LK_GIP_sum"][0][:, dof], [25, 75]),
            )
            print(
                "Delan nMSE: " + "{:.2f}".format(np.median(curr_dict["Delan"][0][:, dof])),
                np.percentile(curr_dict["Delan"][0][:, dof], [25, 75]),
            )

        ax.set_title("joint " + str(dof + 1), fontsize=10)
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_xlim(0.65, 3.35)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        ax.set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
        if dof == 0 or dof == ncols_fig:
            ax.set_ylabel(score + "%")
        ax.grid("minor")
        data = [curr_dict[model][0][:, dof] for model in model_names]
        labels = ["SE", "GIP", "LSE", "LIP", "DeL"]
        x = [1, 1.5, 2, 2.5, 3]
        bplots = ax.boxplot(
            data,
            positions=x,
            widths=0.3,
            patch_artist=True,
            labels=labels,
            showfliers=False,
            medianprops={"color": "red", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        for patch, color in zip(bplots["boxes"], colors):
            patch.set_facecolor(color)
    ax.set_xticks(x, labels)
    fig.tight_layout()

    # gen comulative error boxplots
    data_cum = [np.mean(curr_dict[model][0], axis=1) for model in model_names]
    axes_cum[ndof - 3].set_title(str(dof + 1) + "dof", fontsize=10)
    axes_cum[ndof - 3].set_yscale("log")
    axes_cum[ndof - 3].grid("minor")
    axes_cum[ndof - 3].minorticks_off()
    axes_cum[ndof - 3].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    axes_cum[ndof - 3].set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    axes_cum[ndof - 3].set_xlim(0.65, 3.35)
    if ndof == 3:
        axes_cum[ndof - 3].set_ylabel(score + "%")
    x = [1, 1.5, 2, 2.5, 3]
    bplots_cum = axes_cum[ndof - 3].boxplot(
        data_cum,
        positions=x,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "red", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    for patch, color in zip(bplots_cum["boxes"], colors):
        patch.set_facecolor(color)
    axes_cum[ndof - 3].set_xticklabels(labels, fontsize=8)
    fig_cum.tight_layout()

# data efficiency
# model_names = ["SJ_RBF", "SJ_GIP", "LK_RBF_1_sc", "LK_GIP_sum"]
# de_models_nmse_list = []
# for model in model_names:
#     de_nmse_list = []
#     for num_dat_tr in range(50, 550, 50):
#         file_name = (
#             data_folder
#             + "7dof/"
#             + "de"
#             + str(num_dat_tr)
#             + "_tr_num_sin50_test_num_sin50_seed1_model_"
#             + model
#             + "_test1_estimates.pkl"
#         )
#         data = pkl.load(open(file_name, "rb"))
#         nmse = Project_Utils.get_stat_estimate(
#             Y=data[0]["Y_noiseless"], Y_hat=data[0]["Y_hat"], stat_name="MSE", flg_print=False
#         )
#         de_nmse_list.append(np.sum(nmse))
#     de_nmse_array = np.array(de_nmse_list)
#     de_models_nmse_list.append(de_nmse_array)
#     exec("nmse_dict_" + str(n) + "dof['" + model + "'] = rng_nmse_list")

# fig = plt.figure(figsize=(4.5, 4.5))
# for i, model in enumerate(model_names):
#     plt.plot(range(50, 550, 50), de_models_nmse_list[i], label=labels[i], color=colors[i])
# plt.grid("both")
# plt.yscale("log")
# plt.xlim((50, 500))
# plt.ylabel("Global MSE")
# plt.xlabel("N training Data")
# plt.xticks([i for i in range(50, 550, 50)])
# plt.legend()


T_dict = T_nmse_dict_6dof
V_dict = V_nmse_dict_6dof
model_names = ["LK_GIP_sum", "Delan"]
fig, axes = plt.subplots(2, 1)
fig.set_figheight(4.5)
fig.set_figwidth(3)
ax = axes[0]
ax.set_title("Kinetic Energy", fontsize=10)
ax.set_yscale("log")
ax.minorticks_off()
ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10])
ax.set_yticklabels([0.01, 0.03, 0.1, 0.3, 1, 3, 10])

ax.set_ylabel(score + "%" if score == "nMSE" else score)
ax.grid("minor")
data = [T_dict[model][0][:, 0] * 100 for model in model_names]
labels = ["LIP", "DeL"]
x = [1, 1.5]
bplots = ax.boxplot(
    data,
    positions=x,
    widths=0.3,
    patch_artist=True,
    labels=labels,
    showfliers=False,
    medianprops={"color": "red", "linewidth": 1.0},
    whiskerprops={"color": "black", "linewidth": 1.0},
    capprops={"color": "black", "linewidth": 1.0},
)
for patch, color in zip(bplots["boxes"], ["blue", "purple"]):
    patch.set_facecolor(color)
ax.set_xticks(x, labels)

ax = axes[1]
ax.set_title("Potential Energy", fontsize=10)
ax.set_yscale("log")
ax.minorticks_off()
ax.set_xlim(0.65, 1.85)
ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
ax.set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
ax.set_ylabel(score + "%" if score == "nMSE" else score)
ax.grid("minor")
data = [V_dict[model][0][:, 0] for model in model_names]
labels = ["LIP", "DeL"]
x = [1, 1.5]
bplots = ax.boxplot(
    data,
    positions=x,
    widths=0.3,
    patch_artist=True,
    labels=labels,
    showfliers=False,
    medianprops={"color": "red", "linewidth": 1.0},
    whiskerprops={"color": "black", "linewidth": 1.0},
    capprops={"color": "black", "linewidth": 1.0},
)
for patch, color in zip(bplots["boxes"], ["blue", "purple"]):
    patch.set_facecolor(color)
ax.set_xticks(x, labels)
fig.tight_layout()

plt.show()
