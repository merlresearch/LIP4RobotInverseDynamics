<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Lagrangian Inspired Polynomial Kernel for Robot Inverse Dynamics Learning

This repository contains the source code to learn robot inverse dyanimcs models, based on the Lagrangian Inspired Polunomial (LIP) Kernal described in our IEEE TRO paper
**G. Giacomuzzo, R. Carli, D. Romeres, A. Dalla Libera, 'A Black-Box Physics-Informed Estimator based on Gaussian Process
Regression for Robot Inverse Dynamics Identification'**. The associated data and trained models can be downloaded [here](https://doi.org/10.5281/zenodo.12516500).

## Features

This repository produces the results shown on our paper
G. Giacomuzzo, R. Carli, D. Romeres, A. Dalla Libera, 'A Black-Box Physics-Informed Estimator based on Gaussian Process
Regression for Robot Inverse Dynamics Identification'. Additionally, it can be used to learn the inverse dynamics, the inertia matrix, the coriolis and gravitational contributions of rigid body systems like robot manipulators.

The project directory is organized as follows:

1. Utility functions to read config files from python scripts

    `config_files/Utils.py`

2. Python library used to implement the Gaussian Proccess estimators

    `gpr_lib/`

3. Code for the Monte-Carlo experiment on the simulated PANDA robot described in Section 5.A of the paper.

    ```PANDA_MC_test/```

4. Scripts and config files to learn the inverse dynamics on the real PANDA robot described in Section 5.B.1 of the paper on the MELFA RV4FL robot

    `RealData/PANDA/`

    `RealData/MELFA_RV4FRL`

5. Code used to simulate the PANDA robot using the SymPyBotics library

    `simulated_envs/`

6. Some pytest tests:

    `tests/`

7. Python scripts on the top level:

    `GP_estimator_real_data.py` script to perform GPR with Lagrangian models on real data

    `GP_estimator_single_joint.py` script to perform GPR with single joint approach

    `GP_estimator.py` script to perform GPR with with Lagrangian models

    `Models_Lagrangian_kernel.py` library file containing Lagrangian models implementation based on `gpr_lib`

    `Models.py` library file containing single joint models implementation based on `gpr_lib`

    `Parametric_estimator.py` script to perform parametric identification

    `Project_Utils_.py` utils to load data, compute scores and plot results

## Usage

To replicate the results presented in the paper, after installing this repository, please download the data and the model from [here](https://doi.org/10.5281/zenodo.12516500) and follow the subsequent steps.

### Monte Carlo Experiment on simulated PANDA robot (section 5.A)
Choose one of the following options:
1. Download data and generate the plots
- Download `Simulation_PANDA/Results/` directory from [here](https://doi.org/10.5281/zenodo.12516500)
- Copy the `Results/` directory inside the `PANDA_MC_test/` directory
- Generate plots:
    ```
    cd PANDA_MC_test/plots
    python generate_figures.py
    ```

2. Re-run the experiments using the provided models
- Download `Simulation_PANDA/Results/` directory from [here](https://doi.org/10.5281/zenodo.12516500)
- Copy the `Results/` directory inside the `PANDA_MC_test/` directory
- The directory `PANDA_MC_test/Results/` contains the trained models
- Run test bash script to execute all the experiments:
    ```
    $ cd PANDA_MC_test/
    $ bash ./run_whole_MC_test[_cuda].sh # (use _cuda if want to run the experiment on GPU)
    ```
- The script will generate the test data inside `PANDA_MC_test/data/` and test the models.
    Predictions on test trajectories will be saved in `PANDA_MC_test/Results/`.
    Logs will be saved in `PANDA_MC_test/log/MC_test_[cuda_]log.txt`.

- Generate plots:
    ```
    $ cd PANDA_MC_test/plots
    $ python generate_figures.py
    ```
3. Train and test the models
- Run the train and test bash script:
    ```
    $ cd PANDA_MC_test/
    $ bash ./run_whole_MC_train_test[_cuda].sh # (use cuda if want to run the experiment on GPU)
    ```
- The script will generate the data inside `PANDA_MC_test/data/`, train the models and test their performance.
    Models and predictions on test trajectories will be saved in `PANDA_MC_test/Results/`.
    Logs will be saved in `PANDA_MC_test/log/MC_train_test_[cuda_]log.txt`.

- Generate plots:
    ```
    $ cd PANDA_MC_test/plots
    $ python generate_figures.py
    ```

### Identification of the real PANDA robot:
- Download the `Robots/PANDA/Results` directory from [here](https://doi.org/10.5281/zenodo.12516500)
- Copy the downloaded `Results/` directory inside the `RealData/PANDA/` directory
- Download the `Robots/PANDA/Experiments` directory from [here](https://doi.org/10.5281/zenodo.12516500)
- Copy the downloaded `Experiments/` directory inside the `RealData/PANDA/` directory

1. Plots generation
- `RealData/PANDA/Results` contains the results presented in the paper. To generate plots:
    ```
    $ cd RealData/PANDA/
    $ python plot_nMSE_boxplot.py
    ```

2. Re-run the experiments (train and test)
- Generate PANDA numeric functions:
    ```
    $ cd gpr_lib/GP_prior/LK
    $ python Compute_lagrangian_kernel_LIP.py
    $ python Compute_lagrangian_kernel_LSE.py
    ```
- Generate config files:
    ```
    $ cd RealData/PANDA/config/MC_TEST_REAL
    $ python gen_config.py
    $ python gen_config_single_joint.py

- Run the experiment:
    $ cd ../../ # (move to the RealData/PANDA directory)
    $ bash ./run_PANDA_experiments.sh
    ```
- Generate plots:
    ```
    $ cd RealData/PANDA/CollectedData/
    $ python plot_nMSE_boxplot.py
    ```

### Identification of the MELFA robot:
- Download the `Robots/MELFA_RV4FRL/Results` directory from [here](https://doi.org/10.5281/zenodo.12516500)
- Copy the downloaded `Results/` directory inside the `RealData/MELFA_RV4FRL/` directory
- Download the `Robots/MELFA_RV4FRL/Experiments` directory from [here](https://doi.org/10.5281/zenodo.12516500)
- Copy the downloaded `Experiments/` directory inside the `RealData/MELFA_RV4FRL/` directory

1. Plots generation
- `RealData/MELFA_RV4FRL/Results/` contains the results presented in the paper. To generate plots:
    ```
    $ cd RealData/MELFA_RV4FRL/
    ```
    To print the nMSE boxplots:
    ```
    $ python plot_nMSE_boxplot.py
    ```
    To print the input distribution:
    ```
    $ python plot_input_distribution.py
    ```

2. Re-run the experiments (train and test)
- Generate MELFA numeric functions:
    ```
    $ cd gpr_lib/GP_prior/LK
    $ python Compute_lagrangian_kernel_LIP.py -robot_strucutre '000000' -robot_name 'PANDA_6dof'
    $ python Compute_lagrangian_kernel_LSE.py -robot_strucutre '000000' -robot_name '6dof'
    ```
- Generate config files:
    ```
    $ cd RealData/MELFA/config/MC_TEST_REAL/
    $ python gen_config.py
    $ python gen_config_single_joint.py
    $ python gen_config_parametric_ID.py
    ```
- Run the experiment
    ```
    $ cd ../../ # (move to the RealData/MELFA/ folder)
    $ bash ./run_MELFA_experiments.sh
    ```
- Generate plots:
    ```
    $ cd RealData/PANDA/CollectedData/
    $ python plot_nMSE_boxplot.py
    ```

### Train the LIP estimator on your own dataset

- Assume your data are stored in `data/my_data.pkl`
    - data should be a pandas dataframe with column names `[q_1, ..., q_n, dq_1, ..., dq_n, ddq_1, ..., ddq_n, 'output'_1, .. ,'output'_n]`
        with the output name specified in the config file
- write your own config file, see for example the file generated inside `PANDA_MC_test/config/` for examples of the required parameters. The config file is assumed to be in `config_files/my_config.ini`

    ```
    $ python GP_estimator.py -config 'config_files/my_config.ini' -kernel_name 'LK_GIP_sum'
    ```

## Installation

1. Clone or download this repository.
2. Create a conda environment

```
conda env create --file=environment.yaml
conda activate LIP4RID
pip install torch
# This is not mandatory
pre-commit install
```

## Contacts

Giulio Giacomuzzo: giulio.giacomuzzo@gmail.com

Alberto Dalla Libera: alberto.dallalibera.1@gmail.com

Diego Romeres: romeres@merl.com

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for our policy on contributions.

## License
Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted in the external libraries below:

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

Three external libraries are included in this repository:

- [GPR-PyTorch](https://bitbucket.org/AlbertoDallaLibera/gpr-pytorch/src/master/) with MIT license [LICENSE.md](gpr_lib/LICENSE.md)
- [SymPyBotics](https://github.com/cdsousa/SymPyBotics) with BSD-3-Clause license [LICENSE.txt](simulated_envs/sympybotics/LICENSE.txt)
- [Deep Lagrangian Networks](https://github.com/milutter/deep_lagrangian_networks/tree/main) with MIT license [LICENSE.txt](PANDA_MC_test/deep_lagrangian_networks/LICENSE.txt)
