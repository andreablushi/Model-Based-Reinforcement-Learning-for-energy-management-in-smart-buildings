# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import mbrl.types

from . import Model


class ModelEnvPerBuild:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        models: List[Model],
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_models = models
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = models[0].device
        self.num_buildings = len(models)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        assert isinstance(self.observation_space, gym.spaces.Box), "Serve Box"
        D = self.observation_space.shape[0]
        assert D % self.num_buildings == 0, f"D ({D}) non divisibile per 3"
        subdim = D // self.num_buildings

        # creiamo 3 Box identici con i limiti spezzati
        lows  = np.split(self.observation_space.low,  self.num_buildings)
        highs = np.split(self.observation_space.high, self.num_buildings)
        self.observation_space_per_build = gym.spaces.Tuple([
            gym.spaces.Box(low=l, high=h, dtype=self.observation_space.dtype)
            for l, h in zip(lows, highs)
        ])

        # stessa cosa per action_space se è Box
        assert isinstance(self.action_space, gym.spaces.Box), "Serve Box"
        A = self.action_space.shape[0]
        assert A % self.num_buildings == 0, f"A ({A}) non divisibile per 3"
        lows_a  = np.split(self.action_space.low,  self.num_buildings)
        highs_a = np.split(self.action_space.high, self.num_buildings)
        self.action_space_per_build = gym.spaces.Tuple([
            gym.spaces.Box(low=l, high=h, dtype=self.action_space.dtype)
            for l, h in zip(lows_a, highs_a)
        ])

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True

    def reset(
        self, initial_obs_batch: List[np.ndarray], return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        model_states = []
        for b in range(self.num_buildings):
            if isinstance(self.dynamics_models[b], mbrl.models.OneDTransitionRewardModel):
                assert len(initial_obs_batch[b].shape) == 2  # batch, obs_dim
            with torch.no_grad():
                model_state = self.dynamics_models[b].reset(
                    initial_obs_batch[b].astype(np.float32), rng=self._rng
                )
            self._return_as_np = return_as_np
            model_states.append(model_state if model_state is not None else {})
        return model_states

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, terminated flag and metadata.
            The terminated flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
            ) = self.dynamics_models.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            terminated = self.termination_fn(actions, next_observs)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                terminated = terminated.cpu().numpy()
            return next_observs, rewards, terminated, next_model_state

    def step_plus_gaussians(
            self,
            actions: List[mbrl.types.TensorType],
            model_state: List[Dict[str, torch.Tensor]],
            sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict, mbrl.types.TensorType, mbrl.types.TensorType]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, terminated flag, model state.
            For m2ac also chosen_means, chosen_stds, means_of_all_ensembles, stds_of_all_ensembles, model_indices.
            The done flag is computed using the termination_fn passed in the constructor.
            chosen_means is [model_input.shape[0], observationsize+1] Tensor of means chosen by model_indices
            chosen_stds is [model_input.shape[0], observationsize+1] Tensor of stds chosen by model_indices
            means_of_all_ensembles  is [ensemble_size,model_input.shape[0], observationsize+1] Tensor with
            the mean of every(ensemble_size many) MLP for each observation action pair
            stds_of_all_ensembles  is [ensemble_size,model_input.shape[0], observationsize+1] Tensor with
            the stds of every(ensemble_size many) MLP for each observation action pair
            model_indices is a Tensor[model_input.shape[0]] with random indices [0-ensemble_size) which
            tells which model was chosen for which observation
        """
        assert len(actions[0].shape) == 2  # batch, action_dim
        with torch.no_grad():
            result = []
            for b in range(self.num_buildings):
            # if actions is tensor, code assumes it's already on self.device
                if isinstance(actions, np.ndarray):
                    actions = torch.from_numpy(actions).to(self.device)
                (
                    next_observs,
                    pred_rewards,
                    pred_terminals,
                    next_model_state,
                    chosen_means,
                    chosen_stds,
                    means_of_all_ensembles,
                    stds_of_all_ensembles,
                    model_indices
                ) = self.dynamics_models[b].sample_plus_gaussians(
                    actions[b],
                    model_state[b],
                    deterministic=not sample,
                    rng=self._rng,
                )
                rewards = (
                    pred_rewards
                    if self.reward_fn is None
                    else self.reward_fn(actions[b], next_observs)
                )
                terminated = self.termination_fn(actions[b], next_observs)

                if pred_terminals is not None:
                    raise NotImplementedError(
                        "ModelEnv doesn't yet support simulating terminal indicators."
                    )

                if self._return_as_np:
                    next_observs = next_observs.cpu().numpy()
                    rewards = rewards.cpu().numpy()
                    terminated = terminated.cpu().numpy()
                    chosen_means.cpu().numpy()
                    chosen_stds.cpu().numpy()
                    means_of_all_ensembles.cpu().numpy()
                    stds_of_all_ensembles.cpu().numpy()
                    model_indices.cpu().numpy()
                result.append((next_observs, rewards, terminated, next_model_state,
                        chosen_means, chosen_stds, means_of_all_ensembles,
                        stds_of_all_ensembles, model_indices))
            return result


    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, terminateds, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                rewards[terminated] = 0
                terminated |= terminateds
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)
