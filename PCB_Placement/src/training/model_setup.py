"""
The module for setting up and initializing reinforcement learning models.

This module provides functions to setup and initialize models for reinforcement
learning. Currently, it supports the TD3 (Twin Delayed DDPG) and SAC (Soft
Actor-Critic) models.

Supported models:
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)
- PPO (Proximal Policy Optimization)
- DDPG
- DQN
- A2C

Module Functions:
    - td3_model_setup(train_env, hyperparameters, device="cpu",
        early_stopping: int = 100_000, verbose: int = 0):
        Setup function for the TD3 model.

    - sac_model_setup(train_env, hyperparameters, device="cpu",
        early_stopping: int = 100_000, verbose: int = 0):
        Setup function for the SAC model.

    - setup_model(model_type: str, train_env, hyperparameters,
        device: str = "cpu", early_stopping: int = 100_000, verbose: int = 0):
        Setup function to create a model based on the specified model type.
"""

import TD3
import SAC
import PPO
import DDPG
import DQN
import A2C

supported_models = ["TD3", "SAC", "PPO", "DDPG", "DQN", "A2C"]

def td3_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the TD3 model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the TD3 model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        TD3: The initialized TD3 model.
    """
    model = TD3.TD3(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose)
    return model

def ddpg_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the TD3 model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the TD3 model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        TD3: The initialized TD3 model.
    """
    model = DDPG.DDPG(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose)
    return model

def dqn_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the TD3 model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the TD3 model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        TD3: The initialized TD3 model.
    """
    model = DQN.DQN(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose)
    return model

def ppo_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the TD3 model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the PPO model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        TD3: The initialized TD3 model.
    """
    model = PPO.PPO(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose)
    return model

def a2c_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the TD3 model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the A2C model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        TD3: The initialized TD3 model.
    """
    model = A2C.A2C(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose)
    return model

def sac_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the SAC model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the SAC model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        SAC: The initialized SAC model.
    """
    model = SAC.SAC(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device = device,
                    early_stopping = early_stopping,
                    verbose=verbose)
    return model

def setup_model( model_type: str,
                train_env,
                hyperparameters,
                device:str = "cpu",
                early_stopping:int = 100_000,
                verbose:int = 0 ):
    """
    Setup function to create a model based on the specified model type.

    Args:
        model_type (str): The type of model to setup ("TD3", "SAC" or "PPO").
        train_env: The training environment.
        hyperparameters: Hyperparameters for the model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
            (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
            2: detailed output) (default: 0).

    Returns:
        TD3 or SAC: The initialized model based on the specified model type.

    """
    if model_type == "TD3":
        model = td3_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    elif model_type == "DDPG":
        model = ddpg_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    elif model_type == "DQN":
        model = dqn_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    elif model_type == "SAC":
        model = sac_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    elif model_type == "PPO":           # PPO Model
        model = ppo_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    else:  #A2C Model
        model = a2c_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)



#        print(f"{model_type} is not a supported model.\
#              Please select from {supported_models}")

    return model
