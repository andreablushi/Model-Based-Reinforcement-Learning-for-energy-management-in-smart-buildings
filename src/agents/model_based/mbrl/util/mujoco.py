# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import gymnasium as gym
import gymnasium.wrappers
import numpy as np
import sys
import mbrl.env.mujoco_envs
import mbrl.planning
import mbrl.types
from mbrl.util.env import EnvHandler, Freeze


# Include the mujoco environments in mbrl.env
def _is_mujoco_gym_env(env: gymnasium.wrappers.TimeLimit) -> bool:
    class_module = env.unwrapped.__class__.__module__
    return "gymnasium.envs.mujoco" in class_module or (
        "mbrl.env." in class_module and hasattr(env.env, "data")
    )


class FreezeMujoco(Freeze):
    """Provides a context to freeze a Mujoco environment.

    This context allows the user to manipulate the state of a Mujoco environment and return it
    to its original state upon exiting the context.

    Example usage:

    .. code-block:: python

       env = gym.make("HalfCheetah-v4")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with FreezeMujoco(env):
           step_the_env_a_bunch_of_times()
       o1, *_ = env.step(action) # o1 will be equal to what o1_expected would have been

    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment to freeze.
    """

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if not _is_mujoco_gym_env(env):
            raise RuntimeError(f"Tried to freeze an unsupported environment {env}.")

    def __enter__(self):
        self._init_state = (
            self._env.env.data.qpos.ravel().copy(),
            self._env.env.data.qvel.ravel().copy(),
        )
        self._elapsed_steps = self._env._elapsed_steps

    def __exit__(self, *_args):
        self._env.set_state(*self._init_state)
        self._env._elapsed_steps = self._elapsed_steps


class MujocoEnvHandler(EnvHandler):
    """Env handler for Mujoco-backed gym envs"""

    freeze = FreezeMujoco

    @staticmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        return _is_mujoco_gym_env(env)

    @staticmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        # Handle standard MuJoCo envs
        if "gym___" in env_name:
            env = gym.make(env_name.split("___")[1])
        # Handle custom MuJoco envs in mbrl-lib
        else:
            if env_name == "cartpole_continuous":
                env = mbrl.env.cartpole_continuous.CartPoleEnv()
            elif env_name == "pets_cartpole":
                env = mbrl.env.mujoco_envs.CartPoleEnv()
            elif env_name == "pets_halfcheetah":
                env = mbrl.env.mujoco_envs.HalfCheetahEnv()
            elif env_name == "pets_reacher":
                env = mbrl.env.mujoco_envs.Reacher3DEnv()
            elif env_name == "pets_pusher":
                env = mbrl.env.mujoco_envs.PusherEnv()
            elif env_name == "ant_truncated_obs":
                env = mbrl.env.mujoco_envs.AntTruncatedObsEnv()
            elif env_name == "humanoid_truncated_obs":
                env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv()
            else:
                raise ValueError("Invalid environment string.")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env

    @staticmethod
    def get_current_state(env: gym.Env) -> Tuple:
        """Returns the internal state of the environment, compatible with Gym and CityLearn wrappers.

        Unwraps any TimeLimit or Vectorized wrappers to access the core environment.
        For MuJoCo-based env (e.g., HalfCheetah), returns (qpos, qvel).
        For CityLearnEnv, returns the state array and zero velocities.
        Also returns elapsed steps tracked by TimeLimit if available.

        Args:
            env (gym.Env): potentially wrapped environment.

        Returns:
            ((pos, vel), elapsed_steps):
              - pos, vel: numpy arrays of positions and velocities or state and zeros
              - elapsed_steps: int of steps taken so far, or 0 if unavailable
        """
        # Get elapsed steps if TimeLimit wrapper is present
        elapsed_steps = getattr(env, '_elapsed_steps',
                                getattr(env, 'elapsed_steps', 0))
        # Unwrap until base environment
        base_env = env
        
        # MuJoCo env: expects .data.qpos and .data.qvel
        if hasattr(base_env, 'data') and hasattr(base_env.data, 'qpos'):
            qpos = base_env.data.qpos.ravel().copy()
            qvel = base_env.data.qvel.ravel().copy()
            return (qpos, qvel), elapsed_steps
                # CityLearnEnv: ensure a state-like array from observation_space
        if hasattr(base_env, 'observation_space'):
            # Create a dummy state as zeros vector matching observation space
            obs_shape = base_env.observation_space.shape
            pos = np.zeros(obs_shape, dtype=float).ravel()
            vel = np.zeros_like(pos)
            return (pos, vel), elapsed_steps
        # Unsupported environment
        raise AttributeError(
            f"Cannot extract state from environment of type {type(base_env)}")

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        env.set_state(*state[0])
        env._elapsed_steps = state[1]
