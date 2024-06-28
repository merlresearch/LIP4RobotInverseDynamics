# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import pickle as pkl
import sys
import time

import matplotlib as mp
import numpy as np
import torch

sys.path.insert(0, "../")

try:
    mp.use("Qt5Agg")
    # mp.rc('text', usetex=True)
    # mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

except ImportError:
    pass

import matplotlib.pyplot as plt
from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import init_env_panda, load_dataset_panda

import Project_Utils_ as Project_Utils

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        nargs=1,
        type=int,
        required=False,
        default=[
            True,
        ],
        help="Training using CUDA.",
    )
    parser.add_argument(
        "-i",
        nargs=1,
        type=int,
        required=False,
        default=[
            0,
        ],
        help="Set the CUDA id.",
    )
    parser.add_argument(
        "-s",
        nargs=1,
        type=int,
        required=False,
        default=[
            42,
        ],
        help="Set the random seed",
    )
    parser.add_argument(
        "-r",
        nargs=1,
        type=int,
        required=False,
        default=[
            0,
        ],
        help="Render the figure",
    )
    parser.add_argument(
        "-l",
        nargs=1,
        type=int,
        required=False,
        default=[
            0,
        ],
        help="Load the DeLaN model",
    )
    parser.add_argument(
        "-m",
        nargs=1,
        type=int,
        required=False,
        default=[
            0,
        ],
        help="Save the DeLaN model",
    )
    parser.add_argument(
        "-t",
        nargs=1,
        type=int,
        required=False,
        default=[
            0,
        ],
        help="Save the trajectories",
    )
    parser.add_argument(
        "-f1",
        nargs=1,
        type=str,
        required=False,
        default=[
            "",
        ],
        help="training filename",
    )
    parser.add_argument(
        "-f2",
        nargs=1,
        type=str,
        required=False,
        default=[
            "",
        ],
        help="test filename",
    )
    parser.add_argument(
        "-f1M",
        nargs=1,
        type=str,
        required=False,
        default=[
            "",
        ],
        help="training inertia filename",
    )
    parser.add_argument(
        "-f2M",
        nargs=1,
        type=str,
        required=False,
        default=[
            "",
        ],
        help="test inertia filename",
    )
    parser.add_argument(
        "-n",
        nargs=1,
        type=int,
        required=False,
        default=[
            7,
        ],
        help="number of degrees of freedom",
    )
    parser.add_argument(
        "-saving_path",
        nargs=1,
        type=str,
        required=False,
        default=[
            "./results/",
        ],
        help="saving path",
    )
    parser.add_argument(
        "-model_loading_path",
        nargs=1,
        type=str,
        required=False,
        default=[
            "./results/",
        ],
        help="model loading path",
    )
    parser.add_argument(
        "-downsampling_train",
        nargs=1,
        type=int,
        required=False,
        default=[
            1,
        ],
        help="downsampling training dataset",
    )
    parser.add_argument(
        "-downsampling_test",
        nargs=1,
        type=int,
        required=False,
        default=[
            10,
        ],
        help="downsampling test dataset",
    )
    parser.add_argument(
        "-n_threads",
        nargs=1,
        type=int,
        required=False,
        default=[
            4,
        ],
        help="downsampling test dataset",
    )

    (
        seed,
        cuda,
        render,
        load_model,
        save_model,
        save_trj,
        f1,
        f2,
        f1M,
        f2M,
        n_dof,
        saving_path,
        model_loading_path,
        downampling_train,
        downampling_test,
        n_threads,
    ) = init_env_panda(parser.parse_args())

    # Set saving/loading paths
    estimate_tr_saving_path = saving_path + "Model_Delan_tr_estimates.pkl"
    estimate_test1_saving_path = saving_path + "model_Delan_test1_estimates.pkl"

    # Read the dataset:
    # n_dof = 7
    train_data, test_data = load_dataset_panda(f1, f2, f1M, f2M, n_dof)
    (
        train_qp,
        train_qv,
        train_qa,
        train_tau,
        train_tau_noiseless,
        U_tr,
        M_tr,
        T_tr,
        train_m,
        train_c,
        train_g,
    ) = train_data
    test_qp, test_qv, test_qa, test_tau, test_tau_noiseless, U_test, M_test, T_test, test_m, test_c, test_g = test_data

    # Downsampling -training
    train_qp = train_qp[::downampling_train, :]
    train_qv = train_qv[::downampling_train, :]
    train_qa = train_qa[::downampling_train, :]
    train_tau = train_tau[::downampling_train, :]
    train_tau_noiseless = train_tau_noiseless[::downampling_train, :]
    U_tr = U_tr[::downampling_train]
    M_tr = M_tr[::downampling_train, :, :]
    T_tr = T_tr[::downampling_train]
    train_m = train_m[::downampling_train, :]
    train_c = train_c[::downampling_train, :]
    train_g = train_g[::downampling_train, :]

    # Downsampling test
    test_qp = test_qp[::downampling_test, :]
    test_qv = test_qv[::downampling_test, :]
    test_qa = test_qa[::downampling_test, :]
    test_tau = test_tau[::downampling_test, :]
    test_tau_noiseless = test_tau_noiseless[::downampling_test, :]
    U_test = U_test[::downampling_test]
    M_test = M_test[::downampling_test, :, :]
    T_test = T_test[::downampling_test]
    test_m = test_m[::downampling_test, :]
    test_c = test_c[::downampling_test, :]
    test_g = test_g[::downampling_test, :]

    print("training shape: ", train_qp.shape)
    print("test shape: ", test_qp.shape)

    # Training Parameters:
    # print("\n################################################")
    # print("Training/Loading Deep Lagrangian Networks (DeLaN) {} dofs:".format(n_dof))

    # Construct Hyperparameters:
    hyper = {
        "n_width": 64,
        "n_depth": 2,
        "diagonal_epsilon": 0.01,
        "activation": "SoftPlus",
        "b_init": 1.0e-4,
        "b_diag_init": 0.001,
        "w_init": "xavier_normal",
        "gain_hidden": np.sqrt(2.0),
        "gain_output": 0.1,
        "n_minibatch": 100,
        "learning_rate": 5.0e-04,
        "weight_decay": 1.0e-5,
        "max_epoch": 200000,
    }  # 10000

    # set number of threads
    torch.set_num_threads(n_threads)

    # Load existing model parameters:
    if load_model:
        load_file = model_loading_path + "model_Delan.torch"
        state = torch.load(load_file, map_location="cpu")

        delan_model = DeepLagrangianNetwork(n_dof, **state["hyper"])
        delan_model.load_state_dict(state["state_dict"])
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    else:

        k = 3  # number of training repetitions
        kfold_par_list = []  # list to store the params of different trainings
        kfold_valid_loss = []  # list to store the losses of different trainings
        # k_fold_list = []
        # #Repeated k-fold cross validation:70% of the training dataset is used for training,30% is used for validation
        for j in range(k):

            #     n_data = train_qp.shape[0]
            #     n_samples = int(n_data*1.)
            #     idx_train = np.random.choice(n_data, n_samples, replace=False)
            #     idx_valid = list(set(list(range(n_data))) - set(idx_train))

            #     rkf_train_q, rkf_train_dq, rkf_train_ddq, rfk_train_tau = train_qp[idx_train,:], train_qv[idx_train,:], train_qa[idx_train,:], train_tau[idx_train,:]
            #     rfk_val_q, rfk_val_dq, rfk_val_ddq, rfk_valid_tau =  train_qp[idx_valid,:], train_qv[idx_valid,:], train_qa[idx_valid,:], train_tau[idx_valid,:]

            print("\nFold ", j + 1, " out of ", k)

            # Construct DeLaN:
            delan_model = DeepLagrangianNetwork(n_dof, **hyper)
            delan_model = delan_model.cuda() if cuda else delan_model.cpu()

            # Generate & Initialize the Optimizer:
            optimizer = torch.optim.Adam(
                delan_model.parameters(), lr=hyper["learning_rate"], weight_decay=hyper["weight_decay"], amsgrad=True
            )

            # Generate Replay Memory:
            # mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ))
            # mem = PyTorchReplayMemory(rkf_train_q.shape[0], hyper["n_minibatch"], mem_dim, cuda)
            # mem.add_samples([rkf_train_q, rkf_train_dq, rkf_train_dq, rfk_train_tau])
            mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
            mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
            mem.add_samples([train_qp, train_qv, train_qa, train_tau])

            # Start Training Loop:
            t0_start = time.perf_counter()

            epoch_i = 0
            loss_list = []
            while epoch_i < hyper["max_epoch"]:
                l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
                l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
                l_mem, n_batches = 0.0, 0.0

                for q, qd, qdd, tau in mem:
                    # print('passed')
                    t0_batch = time.perf_counter()

                    # Reset gradients:
                    optimizer.zero_grad()

                    # Compute the Rigid Body Dynamics Model:
                    tau_hat, dEdt_hat = delan_model(q, qd, qdd)

                    # Compute the loss of the Euler-Lagrange Differential Equation:
                    err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
                    l_mean_inv_dyn = torch.mean(err_inv)
                    l_var_inv_dyn = torch.var(err_inv)

                    # Compute the loss of the Power Conservation:
                    dEdt = torch.matmul(qd.view(-1, n_dof, 1).transpose(dim0=1, dim1=2), tau.view(-1, n_dof, 1)).view(
                        -1
                    )
                    err_dEdt = (dEdt_hat - dEdt) ** 2
                    l_mean_dEdt = torch.mean(err_dEdt)
                    l_var_dEdt = torch.var(err_dEdt)

                    # Compute gradients & update the weights:
                    loss = l_mean_inv_dyn + l_mem_mean_dEdt
                    loss.backward()
                    optimizer.step()

                    # Update internal data:
                    n_batches += 1
                    l_mem += loss.item()
                    l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
                    l_mem_var_inv_dyn += l_var_inv_dyn.item()
                    l_mem_mean_dEdt += l_mean_dEdt.item()
                    l_mem_var_dEdt += l_var_dEdt.item()

                    t_batch = time.perf_counter() - t0_batch

                # Update Epoch Loss & Computation Time:
                l_mem_mean_inv_dyn /= float(n_batches)
                l_mem_var_inv_dyn /= float(n_batches)
                l_mem_mean_dEdt /= float(n_batches)
                l_mem_var_dEdt /= float(n_batches)
                l_mem /= float(n_batches)
                loss_list.append(l_mem)
                epoch_i += 1

                if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
                    print("Epoch {:05d}: ".format(epoch_i), end=" ")
                    print("Time = {:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
                    print("Loss = {:.3e}".format(l_mem), end=", ")
                    print(
                        "Inv Dyn = {:.3e} \u00B1 {:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)),
                        end=", ",
                    )
                    print("Power Con = {:.3e} \u00B1 {:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))

            # Store fold result
            kfold_par_list.append(delan_model.state_dict())
            # val_tau_hat, val_dEdt_hat = delan_model(torch.tensor(rfk_val_q, device=q.device, dtype=q.dtype, requires_grad=False),
            #                                         torch.tensor(rfk_val_dq, device=q.device, dtype=q.dtype, requires_grad=False),
            #                                         torch.tensor(rfk_val_ddq, device=q.device, dtype=q.dtype, requires_grad=False))
            # val_tau_err = np.sum((val_tau_hat.detach().numpy()-rfk_valid_tau)**2, axis=1)
            # # kfold_valid_loss.append(np.mean(val_tau_err))
            kfold_valid_loss.append(l_mem)
            # k_fold_list.append({'loss':np.mean(val_tau_err), 'params':delan_model.state_dict()})

        # find the best parameter set
        opt_idx = np.argmin(kfold_valid_loss)
        print("\nloss vector: ", kfold_valid_loss)
        print("index: ", opt_idx)
        state_dict = kfold_par_list[opt_idx]

        delan_model = DeepLagrangianNetwork(n_dof, **hyper)
        delan_model.load_state_dict(state_dict)
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

        # Save the Model:
        if save_model:
            delan_model.cpu()
            torch.save(
                {"epoch": epoch_i, "hyper": hyper, "state_dict": delan_model.state_dict()},
                saving_path + "model_Delan.torch",
            )

    print("\n################################################")
    print("Evaluating DeLaN:")

    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    # plt.figure()

    # Convert NumPy samples to torch:
    q_test = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd_test = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd_test = torch.from_numpy(test_qa).float().to(delan_model.device)
    q_train = torch.from_numpy(train_qp).float().to(delan_model.device)
    qd_train = torch.from_numpy(train_qv).float().to(delan_model.device)
    qdd_train = torch.from_numpy(train_qa).float().to(delan_model.device)
    zeros_tr = torch.zeros_like(q_train).float().to(delan_model.device)
    zeros_test = torch.zeros_like(q_test).float().to(delan_model.device)

    # Compute the torque decomposition:
    with torch.no_grad():
        t0_tr = time.perf_counter()
        delan_tau_tr = delan_model.inv_dyn(q_train, qd_train, qdd_train).cpu().numpy().squeeze()
        delan_g_tr = delan_model.inv_dyn(q_train, zeros_tr, zeros_tr).cpu().numpy().squeeze()
        delan_c_tr = delan_model.inv_dyn(q_train, qd_train, zeros_tr).cpu().numpy().squeeze() - delan_g_tr
        delan_m_tr = delan_model.inv_dyn(q_train, zeros_tr, qdd_train).cpu().numpy().squeeze() - delan_g_tr
        delan_M_tr = delan_model._dyn_model(q_train, qd_train, qdd_train)[1].cpu().numpy()
        delan_T_tr = delan_model._dyn_model(q_train, qd_train, qdd_train)[4].cpu().numpy()
        delan_V_tr = delan_model._dyn_model(q_train, qd_train, qdd_train)[5].cpu().numpy()
        delan_dEdt_tr = delan_model(q_train, qd_train, qdd_train)[1].cpu()
        t_tr = time.perf_counter() - t0_tr

        t0_test = time.perf_counter()
        delan_tau_test = delan_model.inv_dyn(q_test, qd_test, qdd_test).cpu().numpy().squeeze()
        delan_g_test = delan_model.inv_dyn(q_test, zeros_test, zeros_test).cpu().numpy().squeeze()
        delan_c_test = delan_model.inv_dyn(q_test, qd_test, zeros_test).cpu().numpy().squeeze() - delan_g_test
        delan_m_test = delan_model.inv_dyn(q_test, zeros_test, qdd_test).cpu().numpy().squeeze() - delan_g_test
        delan_M_test = delan_model._dyn_model(q_test, qd_test, qdd_test)[1].cpu().numpy()
        delan_T_test = delan_model._dyn_model(q_test, qd_test, qdd_test)[4].cpu().numpy()
        delan_V_test = delan_model._dyn_model(q_test, qd_test, qdd_test)[5].cpu().numpy()
        delan_dEdt_test = delan_model(q_test, qd_test, qdd_test)[1].cpu()
        t_test = time.perf_counter() - t0_test

    t_batch = (time.perf_counter() - t0_batch) / (3.0 * float(test_qp.shape[0]))

    # # Move model to the CPU:
    # delan_model.cpu()

    # # Compute the joint torque using single samples on the CPU. The evaluation is done using only single samples to
    # # imitate the online control-loop. These online computation are performed on the CPU as this is faster for single
    # # samples.

    # delan_tau_tr, delan_dEdt_tr = np.zeros(test_qp.shape), np.zeros((test_qp.shape[0], 1))
    # t0_evaluation = time.perf_counter()
    # for i in range(test_qp.shape[0]):

    #     with torch.no_grad():

    #         # Convert NumPy samples to torch:
    #         q = torch.from_numpy(test_qp[i]).float().view(1, -1)
    #         qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
    #         qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)

    #         # Compute predicted torque:
    #         out = delan_model(q, qd, qdd)
    #         delan_tau[i] = out[0].cpu().numpy().squeeze()
    #         delan_dEdt[i] = out[1].cpu().numpy()

    # t_eval = (time.perf_counter() - t0_evaluation) / float(test_qp.shape[0])

    # Compute Errors:
    train_dEdt = np.sum(train_tau * train_qv, axis=1).reshape((-1, 1))
    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    err_g_train = 1.0 / float(train_qp.shape[0]) * np.sum((delan_g_tr - train_g) ** 2)
    err_g_test = 1.0 / float(test_qp.shape[0]) * np.sum((delan_g_test - test_g) ** 2)
    err_m_train = 1.0 / float(train_qp.shape[0]) * np.sum((delan_m_tr - train_m) ** 2)
    err_m_test = 1.0 / float(test_qp.shape[0]) * np.sum((delan_m_test - test_m) ** 2)
    err_c_train = 1.0 / float(train_qp.shape[0]) * np.sum((delan_c_tr - train_c) ** 2)
    err_c_test = 1.0 / float(test_qp.shape[0]) * np.sum((delan_c_test - test_c) ** 2)
    err_tau_train = 1.0 / float(train_qp.shape[0]) * np.sum((delan_tau_tr - train_tau) ** 2)
    err_tau_test = 1.0 / float(test_qp.shape[0]) * np.sum((delan_tau_test - test_tau) ** 2)
    err_dEdt_train = 1.0 / float(train_qp.shape[0]) * np.sum((delan_dEdt_tr.numpy() - train_dEdt) ** 2)
    err_dEdt_test = 1.0 / float(test_qp.shape[0]) * np.sum((delan_dEdt_test.numpy() - test_dEdt) ** 2)

    print("\n################################################")

    print("Saving Data:")
    d_tr = Project_Utils.get_results_dict(
        Y=train_tau, Y_hat=delan_tau_tr, norm_coef=1.0, Y_noiseless=train_tau_noiseless
    )
    d_test = Project_Utils.get_results_dict(
        Y=test_tau, Y_hat=delan_tau_test, norm_coef=1.0, Y_noiseless=test_tau_noiseless
    )
    d_tr["m"] = train_m
    d_tr["m_hat"] = delan_m_tr
    d_tr["c"] = train_c
    d_tr["c_hat"] = delan_c_tr
    d_tr["g"] = train_g
    d_tr["g_hat"] = delan_g_tr

    d_test["m"] = test_m
    d_test["m_hat"] = delan_m_test
    d_test["c"] = test_c
    d_test["c_hat"] = delan_c_test
    d_test["g"] = test_g
    d_test["g_hat"] = delan_g_test

    d_tr["T"] = T_tr
    d_tr["T_hat"] = delan_T_tr
    U_offset_tr = U_tr[0] - delan_V_tr[0]
    d_tr["U"] = U_tr
    d_tr["U_hat"] = delan_V_tr + U_offset_tr
    d_tr["L"] = T_tr - U_tr
    d_tr["L_hat"] = delan_T_tr - delan_V_tr

    d_test["T"] = T_test
    d_test["T_hat"] = delan_T_test
    U_offset_test = U_test[0] - delan_V_test[0]
    d_test["U"] = U_test
    d_test["U_hat"] = delan_V_test + U_offset_test
    d_test["L"] = T_test - U_test
    d_test["L_hat"] = delan_T_test - delan_V_test

    if save_trj:
        pkl.dump([d_test, M_test, delan_M_test], open(estimate_test1_saving_path, "wb"))
        print("Data Saved in ", estimate_test1_saving_path)

    print("\n################################################")
    print("Performance on Training Dataset:")
    print(
        "               Torque nMSE = ",
        Project_Utils.get_stat_estimate(Y=d_tr["Y_noiseless"], Y_hat=d_tr["Y_hat"], stat_name="nMSE", flg_print=False),
    )
    print("                Torque MSE = {:.3e}".format(err_tau_train))
    print("              Inertial MSE = {:.3e}".format(err_m_train))
    print("Coriolis & Centrifugal MSE = {:.3e}".format(err_c_train))
    print("         Gravitational MSE = {:.3e}".format(err_g_train))
    print("    Power Conservation MSE = {:.3e}".format(err_dEdt_train))
    print("              Elapsed time = {:.3e}s".format(t_tr))

    # print("\n################################################")

    print("\nPerformance on Test Dataset:")
    print(
        "               Torque nMSE = {}".format(
            Project_Utils.get_stat_estimate(
                Y=d_test["Y_noiseless"], Y_hat=d_test["Y_hat"], stat_name="nMSE", flg_print=False
            )
        )
    )
    print("                Torque MSE = {:.3e}".format(err_tau_test))
    print("              Inertial MSE = {:.3e}".format(err_m_test))
    print("Coriolis & Centrifugal MSE = {:.3e}".format(err_c_test))
    print("         Gravitational MSE = {:.3e}".format(err_g_test))
    print("    Power Conservation MSE = {:.3e}".format(err_dEdt_test))
    print("              Elapsed time = {:.3e}s".format(t_test))

    # print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))

    print("\n################################################")
    if render:
        fig_rows = n_dof // 2 + n_dof % 2
        print("Plotting Results:")
        fig = plt.figure(figsize=(9, 6))
        # plt.suptitle(title)
        for i in range(0, n_dof):
            plt.subplot(fig_rows, 2, i + 1)
            plt.plot(train_tau_noiseless[:, i], "r", label="True")  # =output_name+'_'+str(joint_index_list[i]))
            plt.plot(delan_tau_tr[:, i], "b", label="Delan")  # output_name+'_'+str(joint_index_list[i])+'_hat')
            # plt.ylabel(r'$\tau_'+str(joint_index_list[i])+'$')
            plt.grid()
            plt.legend()
        fig.tight_layout()

        fig = plt.figure(figsize=(9, 6))
        # plt.suptitle(title)
        for i in range(0, n_dof):
            plt.subplot(fig_rows, 2, i + 1)
            plt.plot(test_tau_noiseless[:, i], "r", label="True")  # =output_name+'_'+str(joint_index_list[i]))
            plt.plot(delan_tau_test[:, i], "b", label="Delan")  # output_name+'_'+str(joint_index_list[i])+'_hat')
            # plt.ylabel(r'$\tau_'+str(joint_index_list[i])+'$')
            plt.grid()
            plt.legend()
        fig.tight_layout()

        if not load_model:
            plt.figure()
            plt.plot(loss_list)

        plt.show()

        print("\n################################################")

    # plt.figure()
    # plt.plot(d_test['U'], label = 'True')
    # plt.plot(d_test['U_hat'], label = 'Delan')
    # plt.legend()

    # fig_rows = n_dof//2+n_dof%2
    # fig = plt.figure(figsize=(9, 6))
    # # plt.suptitle(title)
    # for i in range(0,n_dof):
    #     plt.subplot(fig_rows,2,i+1)
    #     plt.plot(test_tau_noiseless[:,i], 'r', label='True')#=output_name+'_'+str(joint_index_list[i]))
    #     plt.plot(delan_tau_test[:,i], 'b', label='Delan')#output_name+'_'+str(joint_index_list[i])+'_hat')
    #     #plt.ylabel(r'$\tau_'+str(joint_index_list[i])+'$')
    #     plt.grid()
    #     plt.legend()
    # fig.tight_layout()

    # plt.show()
