
from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gymnasium as gym
import gymnasium.wrappers
import hydra
import numpy as np
import omegaconf
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedSpaceWrapper, StableBaselines3Wrapper
from citylearn.data import DataSet
from mbrl.util.CityLearnWrappers import CityLearnKPIWrapper, CityLearnWandbWrapper
from mbrl.rewards.CityLearnReward import SolarPenaltyAndComfortReward
from mbrl.rewards.FactorizedCityLearnReward import FactorizedSolarPenaltyAndComfortReward

import mbrl.planning
import mbrl.types

class CityLearnSchema:
    def __init__(self, schema: Dict[str, Any] | None=None):
        self._schema: Dict[str, Any] | None = schema

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema
    
    @schema.setter
    def schema(self, new_schema: Dict[str, Any]):
        self._schema = new_schema

    def load(self, dataset: str, custom: bool=False):
        assert self._schema is None, 'Schema has already been loaded.'

        # Get the schema of the dataset
        self._schema = DataSet().get_schema(dataset)

        if custom:
            print("="*40)
            print("CityLearnOmnisafe - Dataset Customization")
            print("="*40)
            print(f"Dataset: {dataset}")

            # User's building selection
            _ = self._select_items(key='buildings')
            # User's observation selection
            selected_obs = self._select_items(key='observations')
            # User's action selection
            selected_act = self._select_items(key='actions')

            # Sanity check
            self._check(selected_obs, selected_act)
        else:
            # Include Building_1 by default
            self.set_active(key='buildings', items=['Building_1'])
            self._schema['observations']['cooling_electricity_consumption']['active'] = True
            self._schema['observations']['dhw_electricity_consumption']['active'] = True


    def save(self, dir: str, prefix: str='base'):
        with open(f'{dir}/{prefix}_schema.json', 'w') as f:
            json.dump(self._schema, f, indent=4)

    def set(self, key: str, value: Dict[str, Any]):
        self._schema[key] = value

    def set_active(self, key: str, items: List[str]):
        assert self._schema is not None, 'Schema has not been loaded, yet.'
        assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'

        # Filter CityLearn items
        flag_key = 'include' if key == 'buildings' else 'active'
        for it in self._schema[key].keys():
            self._schema[key][it][flag_key] = (it in items)

    def train_test_split(self, frac: float, mode: str):
        assert mode in ['train', 'test'], f'Unknown mode {mode}. Must be either `train` or `test`.'
        assert 0 < frac <= 1, f'Invalid fraction {frac}. Must be in (0,1).'

        # Copy base schema
        train_schema, test_schema = self._schema.copy(), self._schema.copy()

        # Total simulation days
        time_steps = self._schema['simulation_end_time_step'] + 1
        total_days = time_steps // 24

        # Train/test split index
        train_days = int(total_days * frac)
        split_idx = train_days * 24

        # Modify train/test schemas
        train_schema['simulation_end_time_step'] = split_idx - 1
        if frac < 1:
            test_schema['simulation_start_time_step'] = split_idx

        return train_schema, test_schema

    def _select_items(self, key: str):
        assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'
    
        # Available items
        if key == 'buildings':
            pool = list(self._schema[key].keys())
        else:
            pool = [item for item in self._schema[key].keys() if self._schema[key][item]['active']]

        print(f"Available {key}:")
        for idx, item in enumerate(pool):
            print(f"- {idx+1}. {item}")

        # Item selection
        user_input = input(f"\nSelect {item} by entering their numbers separated by commas (e.g., 1,3,5): ")
        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip().isdigit() and 0 < int(i.strip()) <= len(pool)]
        selected_items = [pool[i] for i in selected_indices]

        print(f"Selected items: {selected_items}\n\n")

        # Modify schema according to user's selection
        self.set_active(key=key, items=selected_items)

        return selected_items
    
    def _check(self, observations: List[str], actions: List[str]):
        print('Checking observations...')
        if 'indoor_dry_bulb_temperature' in observations:
            if 'indoor_dry_bulb_temperature_cooling_set_point':
                # Remove "redundant" observations
                observations.remove('indoor_dry_bulb_temperature_cooling_set_point')

                # Activate temperature delta
                observations.append('indoor_dry_bulb_temperature_cooling_delta')
                self.set_active(key='observations', items=observations)
                print(
                    '[CHECK] Both `indoor_dry_bulb_temperature` and `indoor_dry_bulb_temperature_cooling_set_point` are active.' + 
                    ' `indoor_dry_bulb_temperature_cooling_delta` has been activated.'
                )

            if 'indoor_dry_bulb_temperature_heating_set_point' in observations:
                # Remove "reduntant" observations
                observations.remove('indoor_dry_bulb_temperature_heating_set_point')

                # Activate temperature delta
                observations.append('indoor_dry_bulb_temperature_heating_delta')
                self.set_active(key='observations', items=observations)
                print(
                    '[CHECK] Both `indoor_dry_bulb_temperature` and `indoor_dry_bulb_temperature_heating_set_point` are active.' + 
                    ' `indoor_dry_bulb_temperature_heating_delta` has been activated.'
                )


def _get_term_and_reward_fn(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
) -> Tuple[mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    import mbrl.env

    term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
    if hasattr(cfg.overrides, "reward_fn") and cfg.overrides.reward_fn is not None:
        print(f"Using reward function {cfg.overrides.reward_fn}")
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
    else:
        reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn, None)

    return term_fn, reward_fn


def _handle_learned_rewards_and_seed(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    env: gym.Env,
    reward_fn: mbrl.types.RewardFnType,
) -> Tuple[gym.Env, mbrl.types.RewardFnType]:
    if cfg.overrides.get("learned_rewards", True):
        reward_fn = None

    if cfg.seed is not None:
        env.reset(seed=cfg.seed)
        if isinstance(env.action_space, list):
            for action_space in env.action_space:
                action_space.seed(cfg.seed + 2)
            for observation_space in env.observation_space:
                observation_space.seed(cfg.seed + 1)
        else:
            env.observation_space.seed(cfg.seed + 1)
            env.action_space.seed(cfg.seed + 2)

    return env, reward_fn


def _legacy_make_env(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],test_env:bool
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    render_mode = "human" if cfg.get("render", False) else None
    if test_env:
        render_mode = None
    if "dmcontrol___" in cfg.overrides.env:
        import mbrl.third_party.dmc2gym as dmc2gym

        domain, task = cfg.overrides.env.split("___")[1].split("--")
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
        env = dmc2gym.make(domain_name=domain, task_name=task)
        env = gym.make("GymV26Environment-v0", env=env)

    elif "gym___" in cfg.overrides.env:
        env = gym.make(cfg.overrides.env.split("___")[1], render_mode=render_mode)
        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
    else:
        import mbrl.env.mujoco_envs
        import mbrl.env

        if cfg.overrides.env == "cartpole_continuous":
            env = mbrl.env.cartpole_continuous.CartPoleEnv(render_mode=render_mode)
            term_fn = mbrl.env.termination_fns.cartpole
            reward_fn = mbrl.env.reward_fns.cartpole
        elif cfg.overrides.env == "ant_truncated_obs":
            env = mbrl.env.mujoco_envs.AntTruncatedObsEnv(render_mode=render_mode)
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
        elif cfg.overrides.env == "humanoid_truncated_obs":
            env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv(render_mode=render_mode)
            term_fn = mbrl.env.termination_fns.humanoid
            reward_fn = None
        elif cfg.overrides.env == "citylearn":
            schema_name = 'citylearn_challenge_2023_phase_1'
            schema_obj = CityLearnSchema()
            schema_obj.load(dataset=schema_name, custom=False)
            train_schema, test_schema = schema_obj.train_test_split(frac=0.8, mode='train')
            env = CityLearnEnv(
                schema=train_schema, 
                central_agent=True,
            )
            reward_fn = SolarPenaltyAndComfortReward(env.schema)
            env.reward_function = reward_fn

            # env = NormalizedSpaceWrapper(env)
            if not test_env:
                print("HERE")
                env = CityLearnWandbWrapper(env, online=True, verbose=True)
            else:
                env = CityLearnEnv(
                    schema=test_schema, 
                    central_agent=True,
                )
                reward_fn = SolarPenaltyAndComfortReward(env.schema)
                env.reward_function = reward_fn
                env = CityLearnKPIWrapper(env)

            term_fn = mbrl.env.termination_fns.no_termination
        elif cfg.overrides.env == "test_citylearn":
            schema_name = 'citylearn_challenge_2023_phase_1'
            schema_obj = CityLearnSchema()
            schema_obj.load(dataset=schema_name, custom=False)
            schema_obj.set_active(key='buildings', items=[f'Building_2'])

            env = CityLearnEnv(
                schema=schema_obj.schema, 
                central_agent=True,
            )
            reward_fn = SolarPenaltyAndComfortReward(env.schema)
            env.reward_function = reward_fn
            term_fn = mbrl.env.termination_fns.no_termination
            
        else:
            raise ValueError("Invalid environment string.")
        # env = gym.wrappers.TimeLimit(
        #     env, max_episode_steps=cfg.overrides.get("trial_length", 1000)
        # )

    env, _ = _handle_learned_rewards_and_seed(cfg, env, reward_fn)
    return env, term_fn, reward_fn


class Freeze(ABC):
    """Abstract base class for freezing various gym backends"""

    def __enter__(self, env):
        raise NotImplementedError

    def __exit__(self, env):
        raise NotImplementedError


class EnvHandler(ABC):
    """Abstract base class for handling various gym backends

    Subclasses of EnvHandler should define an associated Freeze subclass
    and override self.freeze with that subclass
    """

    freeze = Freeze

    @staticmethod
    @abstractmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        """Checks that the env being handled is of the correct type"""
        raise NotImplementedError

    @staticmethod
    def make_env(
        cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig],
        test_env: bool
    ) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
        """Creates an environment from a given OmegaConf configuration object.

        This method expects the configuration, ``cfg``,
        to have the following attributes (some are optional):

            - If ``cfg.overrides.env_cfg`` is present, this method
            instantiates the environment using `hydra.utils.instantiate(env_cfg)`.
            Otherwise, it expects attribute ``cfg.overrides.env``, which should be a
            string description of the environment where valid options are:

            - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
                with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
            - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
            - "cartpole_continuous": a continuous version of gym's Cartpole environment.
            - "pets_halfcheetah": the implementation of HalfCheetah used in Chua et al.,
                PETS paper.
            - "ant_truncated_obs": the implementation of Ant environment used in Janner et al.,
                MBPO paper.
            - "humanoid_truncated_obs": the implementation of Humanoid environment used in
                Janner et al., MBPO paper.

            - ``cfg.overrides.term_fn``: (only for dmcontrol and gym environments) a string
            indicating the environment's termination function to use when simulating the
            environment with the model. It should correspond to the name of a function in
            :mod:`mbrl.env.termination_fns`.
            - ``cfg.overrides.reward_fn``: (only for dmcontrol and gym environments)
            a string indicating the environment's reward function to use when simulating the
            environment with the model. If not present, it will try to use
            ``cfg.overrides.term_fn``.
            If that's not present either, it will return a ``None`` reward function.
            If provided, it should correspond to the name of a function in
            :mod:`mbrl.env.reward_fns`.
            - ``cfg.overrides.learned_rewards``: (optional) if present indicates that
            the reward function will be learned, in which case the method will return
            a ``None`` reward function.
            - ``cfg.overrides.trial_length``: (optional) if presents indicates the maximum length
            of trials. Defaults to 1000.

        Args:
            cfg (omegaconf.DictConf): the configuration to use.

        Returns:
            (tuple of env, termination function, reward function): returns the new environment,
            the termination function to use, and the reward function to use (or ``None`` if
            ``cfg.learned_rewards == True``).
        """
        # Handle the case where cfg is a dict
        cfg = omegaconf.OmegaConf.create(cfg)
        env_cfg = cfg.overrides.get("env_cfg", None)
        if env_cfg is None:
            return _legacy_make_env(cfg, test_env=test_env)

        env = hydra.utils.instantiate(env_cfg)

        # env = gym.wrappers.TimeLimit(
        #     env, max_steps = cfg.overrides.get("trial_length", 1000)
        # )

        term_fn, reward_fn = _get_term_and_reward_fn(cfg)
        env, reward_fn = _handle_learned_rewards_and_seed(cfg, env, reward_fn)
        return env, term_fn, reward_fn

    @staticmethod
    @abstractmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        """Creates a new environment from its string description.

        Args:
            env_name (str): the string description of the environment.

        Returns:
            (gym.Env): the created environment.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        """Returns the internal state of the environment.

        Returns a tuple with information that can be passed to :func:set_env_state` to manually
        set the environment (or a copy of it) to the same state it had when this function was
        called.

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        """Sets the state of the environment.

        Assumes ``state`` was generated using :func:`get_current_state`.

        Args:
            state (tuple): see :func:`get_current_state` for a description.
            env (:class:`gym.wrappers.TimeLimit`): the environment.
        """
        raise NotImplementedError

    def rollout_env(
        self,
        env: gym.wrappers.TimeLimit,
        initial_obs: np.ndarray,
        lookahead: int,
        agent: Optional[mbrl.planning.Agent] = None,
        plan: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the environment for some number of steps then returns it to its original state.

        Works with mujoco gym and dm_control environments
        (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).

        Args:
            env (:class:`gym.wrappers.TimeLimit`): the environment.
            initial_obs (np.ndarray): the latest observation returned by the environment (only
                needed when ``agent is not None``, to get the first action).
            lookahead (int): the number of steps to run. If ``plan is not None``,
                it is overridden by `len(plan)`.
            agent (:class:`mbrl.planning.Agent`, optional): if given, an agent to obtain actions.
            plan (sequence of np.ndarray, optional): if given, a sequence of actions to execute.
                Takes precedence over ``agent`` when both are given.

        Returns:
            (tuple of np.ndarray): the observations, rewards, and actions observed, respectively.

        """
        actions = []
        real_obses = []
        rewards = []
        with self.freeze(cast(gym.wrappers.TimeLimit, env)):  # type: ignore
            current_obs = initial_obs.copy()
            real_obses.append(current_obs)
            if plan is not None:
                lookahead = len(plan)
            for i in range(lookahead):
                a = plan[i] if plan is not None else agent.act(current_obs)
                if isinstance(a, torch.Tensor):
                    a = a.numpy()
                next_obs, reward, termianted, truncated, _ = env.step(a)
                actions.append(a)
                real_obses.append(next_obs)
                rewards.append(reward)
                if terminated or truncated:
                    break
                current_obs = next_obs
        return np.stack(real_obses), np.stack(rewards), np.stack(actions)