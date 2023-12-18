# PORelDICE

This is the implementation for our paper **Relaxed Stationary Distribution Correction Estimation for Improved Offline Policy Optimization** in Jax.

This codebase is built upon [IQL](https://github.com/ikostrikov/implicit_q_learning.git) and [SQL](https://github.com/ryanxhr/IVR.git) repository.

## Installations

    $ conda create -c nvidia -n PORelDICE python=3.8 cuda-nvcc=11.3
    $ conda activate PORelDICE
    $ pip install -r requirements.txt

## Run Experiments 

Mujoco

    $ ./run_mujoco.sh

Antmaze

    $ ./run_antmaze.sh

Kitchen

    $ ./run_kitchen.sh

  
