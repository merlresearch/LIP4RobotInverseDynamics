import dill as pickle
import numpy as np
import torch


def init_env(args):

    # Set the NumPy Formatter:
    np.set_printoptions(
        suppress=True, precision=2, linewidth=500, formatter={"float_kind": lambda x: "{:+08.2f}".format(x)}
    )

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model = bool(args.r[0]), bool(args.l[0]), bool(args.m[0])

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model


def init_env_panda(args):

    # Set the NumPy Formatter:
    # np.set_printoptions(suppress=True, precision=2, linewidth=500,
    #                     formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model, save_trj = bool(args.r[0]), bool(args.l[0]), bool(args.m[0]), bool(args.t[0])
    f1, f2, ndof = args.f1[0], args.f2[0], args.n[0]
    f1M, f2M = args.f1M[0], args.f2M[0]
    saving_path, model_loading_path = args.saving_path[0], args.model_loading_path[0]
    downsampling_train = args.downsampling_train[0]
    downsampling_test = args.downsampling_test[0]
    n_threads = args.n_threads[0]

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return (
        seed,
        cuda_flag,
        render,
        load_model,
        save_model,
        save_trj,
        f1,
        f2,
        f1M,
        f2M,
        ndof,
        saving_path,
        model_loading_path,
        downsampling_train,
        downsampling_test,
        n_threads,
    )


def init_env_panda_grid_search(args):

    # Read the parameters:
    cuda_id, cuda_flag = args.i[0], args.c[0]
    save_model = bool(args.m[0])
    f1, f2, ndof = args.f1[0], args.f2[0], args.n[0]
    f1M, f2M = args.f1M[0], args.f2M[0]
    saving_path = args.saving_path[0]
    downsampling_train = args.downsampling_train[0]
    downsampling_test = args.downsampling_test[0]
    n_threads = args.n_threads[0]

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # # Set the seed:
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return cuda_flag, save_model, f1, f2, f1M, f2M, ndof, saving_path, downsampling_train, downsampling_test, n_threads


def load_dataset(n_characters=3, filename="data/DeLaN_Data.pickle"):

    with open(filename, "rb") as f:
        data = pickle.load(f)

    n_dof = 2

    dt = np.concatenate([t[1:] - t[:-1] for t in data["t"]])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.0e-12

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_char]

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = (
        np.zeros((0, n_dof)),
        np.zeros((0, n_dof)),
        np.zeros((0, n_dof)),
        np.zeros((0, n_dof)),
    )
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = (
        np.zeros((0, n_dof)),
        np.zeros((0, n_dof)),
        np.zeros((0, n_dof)),
        np.zeros((0, n_dof)),
    )
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [
        0,
    ]  # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (
        (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau),
        (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),
        divider,
        dt_mean,
    )


def load_dataset_panda(
    filename_train="data/FE_panda/FE_panda_pybul_tr.pkl",
    filename_test="data/FE_panda/FE_panda_pybul_test.pkl",
    filename_M_train="",
    filename_M_test="",
    ndofs=7,
):

    data_train = pickle.load(open(filename_train, "rb"))
    data_M_train = pickle.load(open(filename_M_train, "rb"))
    data_test = pickle.load(open(filename_test, "rb"))
    data_M_test = pickle.load(open(filename_M_test, "rb"))

    # n_dof = 7
    qp_labels = ["q_" + str(n) for n in range(1, ndofs + 1)]
    qv_labels = ["dq_" + str(n) for n in range(1, ndofs + 1)]
    qa_labels = ["ddq_" + str(n) for n in range(1, ndofs + 1)]
    tau_labels = ["tau_" + str(n) for n in range(1, ndofs + 1)]
    m_labels = ["m_" + str(n) for n in range(1, ndofs + 1)]
    c_labels = ["c_" + str(n) for n in range(1, ndofs + 1)]
    g_labels = ["g_" + str(n) for n in range(1, ndofs + 1)]
    U_name = ["U"]
    tau_noiseless_labels = ["tau_noiseless_" + str(n) for n in range(1, ndofs + 1)]

    # qv_labels = ['dq_1', 'dq_2', 'dq_3', 'dq_4', 'dq_5', 'dq_6', 'dq_7']
    # qa_labels = ['ddq_1', 'ddq_2', 'ddq_3', 'ddq_4', 'ddq_5', 'ddq_6', 'ddq_7']
    # tau_labels = ['tau_1', 'tau_2', 'tau_3', 'tau_4', 'tau_5', 'tau_6', 'tau_7']
    # tau_noiseless_labels = ['tau_noiseless_1', 'tau_noiseless_2', 'tau_noiseless_3', 'tau_noiseless_4', 'tau_noiseless_5', 'tau_noiseless_6', 'tau_noiseless_7']
    train_qp, train_qv, train_qa, train_tau, train_tau_noiseless = (
        data_train[qp_labels].to_numpy(),
        data_train[qv_labels].to_numpy(),
        data_train[qa_labels].to_numpy(),
        data_train[tau_labels].to_numpy(),
        data_train[tau_noiseless_labels].to_numpy(),
    )
    test_qp, test_qv, test_qa, test_tau, test_tau_noiseless = (
        data_test[qp_labels].to_numpy(),
        data_test[qv_labels].to_numpy(),
        data_test[qa_labels].to_numpy(),
        data_test[tau_labels].to_numpy(),
        data_test[tau_noiseless_labels].to_numpy(),
    )

    m_train, c_train, g_train = (
        data_train[m_labels].to_numpy(),
        data_train[c_labels].to_numpy(),
        data_train[g_labels].to_numpy(),
    )
    m_test, c_test, g_test = (
        data_test[m_labels].to_numpy(),
        data_test[c_labels].to_numpy(),
        data_test[g_labels].to_numpy(),
    )

    U_tr = data_train[U_name].to_numpy().reshape(-1, 1)
    U_test = data_test[U_name].to_numpy().reshape(-1, 1)

    M_tr = np.stack(data_M_train, axis=0)
    M_test = np.stack(data_M_test, axis=0)

    T_tr = 0.5 * np.array(
        [train_qv[i, :].T @ M_tr[i, :, :] @ train_qv[i, :] for i in range(train_qp.shape[0])]
    ).reshape(-1, 1)
    T_test = 0.5 * np.array(
        [test_qv[i, :].T @ M_test[i, :, :] @ test_qv[i, :] for i in range(test_qp.shape[0])]
    ).reshape(-1, 1)

    return (
        train_qp,
        train_qv,
        train_qa,
        train_tau,
        train_tau_noiseless,
        U_tr,
        M_tr,
        T_tr,
        m_train,
        c_train,
        g_train,
    ), (test_qp, test_qv, test_qa, test_tau, test_tau_noiseless, U_test, M_test, T_test, m_test, c_test, g_test)
