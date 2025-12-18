from collections import deque
from citylearn.wrappers import DiscreteActionWrapper
from gymnasium import ActionWrapper, spaces

import itertools

from typing import List, Mapping

from citylearn.citylearn import CityLearnEnv, EvaluationCondition
from citylearn.wrappers import StableBaselines3Wrapper

# Logging
import wandb

class DiscretizeActionWrapper(ActionWrapper):
    """Action wrapper for :py:class:`citylearn.agents.q_learning.TabularQLearning` agent.

    Wraps `env` in :py:class:`citylearn.wrappers.DiscreteActionWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None) -> None:
        env = DiscreteActionWrapper(env, bin_sizes=bin_sizes, default_bin_size=default_bin_size)
        super().__init__(env)
        self.env: CityLearnEnv
        self.combinations = self.set_combinations()

    @property
    def action_space(self) -> List[spaces.Discrete]:
        """Returns action space for discretized actions."""

        action_space = []

        for c in self.combinations:
            action_space.append(spaces.Discrete(len(c)))
        
        return action_space
    
    def action(self, actions: List[float]) -> List[List[int]]:
        """Returns discretized actions."""
        # su github list(c[a[0]])
        return [list(c[a]) for a, c in zip(actions, self.combinations)]
    
    def set_combinations(self) -> List[List[int]]:
        """Returns all combinations of discrete actions."""

        combs_list = []

        for s in self.env.action_space:
            options = [list(range(d.n)) for d in s]
            combs = list(itertools.product(*options))
            combs_list.append(combs)

        return combs_list
    
class CityLearnKPIWrapper(StableBaselines3Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Current episode tracking
        self.current_ep_reward = 0.0
        self.current_ep_length = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_ep_reward += reward
        self.current_ep_length += 1

        if terminated or truncated:
            # Get KPIs
            kpis = self.env.unwrapped.evaluate(
                control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
                baseline_condition=EvaluationCondition.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV,
                comfort_band=2.0,
            )

            # names of KPIs to retrieve from evaluate function
            kpi_names = {
                'cost_total': 'Cost',
                'carbon_emissions_total': 'Emissions',
                'daily_peak_average': 'Avg. daily peak',
                'ramping_average': 'Ramping',
                'monthly_one_minus_load_factor_average': '1 - load factor',
                'discomfort_proportion': 'Discomfort'
            }
            kpis = kpis[
                (kpis['cost_function'].isin(kpi_names))
            ].dropna()
            kpis['cost_function'] = kpis['cost_function'].map(lambda x: kpi_names[x])

            # round up the values to 2 decimal places for readability
            kpis['value'] = kpis['value'].round(2)

            # rename the column that defines the KPIs
            kpis = kpis.rename(columns={'cost_function': 'kpi'})

            # Populate info dict
            for _, row in kpis.iterrows():
                info[row['kpi']] = row['value']
            info['episode_length'] = self.current_ep_length
            info['episode_return'] = self.current_ep_reward
        
        return obs, reward, terminated, truncated, info
            
    def reset(self, **kwargs):
        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        return super().reset(**kwargs)

        


class CityLearnWandbWrapper(StableBaselines3Wrapper):
    """
    Environment wrapper that extracts KPIs at episode end, maintains rolling averages,
    and logs metrics to Weights & Biases.
    
    Parameters
    ----------
    env : CityLearnEnv
        CityLearn environment to wrap.
    online : bool, default=False
        Whether to log data to W&B online.
    window_len : int, default=100
        Length of the rolling window for averaging metrics.
    verbose : bool, default=False
        Whether to print episode summaries.
    """
    
    def __init__(self, env, online: bool = False, window_len: int = 100, verbose: bool = False):
        super().__init__(env)
        
        # W&B logging config
        self.online = online
        self.verbose = verbose
        
        # Episode counter
        self.ep_count = 0
        
        # Rolling windows for metrics
        self.ep_rewards = deque(maxlen=window_len)
        self.ep_lengths = deque(maxlen=window_len)
        self.discomfort_h = deque(maxlen=window_len)
        self.carbon_emissions_h = deque(maxlen=window_len)
        self.net_consumption_h = deque(maxlen=window_len)
        
        # Current episode tracking
        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        
    def _log_fn(self, x):
        """Calculate average of values in deque."""
        return sum(x) / len(x) if len(x) > 0 else 0.0
    
    def reset(self, **kwargs):
        """Reset environment and episode tracking."""
        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        """
        Execute action and track metrics. At episode end, extract KPIs,
        update rolling averages, and log to W&B.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track current episode metrics
        self.current_ep_reward += reward
        self.current_ep_length += 1

        if terminated or truncated:
            # Increment episode counter
            self.ep_count += 1
            
            # Get KPIs from environment
            kpis = self.env.unwrapped.evaluate()
            
            # Filter district level KPIs
            kpis = kpis[kpis['level'] == 'district']
            discomfort = kpis[(kpis['cost_function'] == 'discomfort_proportion')]['value'].item()
            carbon_emissions = kpis[(kpis['cost_function'] == 'carbon_emissions_total')]['value'].item()
            net_consumption = kpis[(kpis['cost_function'] == 'electricity_consumption_total')]['value'].item()
            
            # Populate info dict with current episode KPIs
            info['discomfort'] = discomfort
            info['carbon_emissions'] = carbon_emissions
            info['net_consumption'] = net_consumption
            info['episode'] = {
                'r': self.current_ep_reward,
                'l': self.current_ep_length
            }
            
            # Update rolling windows
            # Update rolling windows
            self.ep_rewards.append(self.current_ep_reward)
            self.ep_lengths.append(self.current_ep_length)
            self.discomfort_h.append(discomfort)
            self.carbon_emissions_h.append(carbon_emissions)
            self.net_consumption_h.append(net_consumption)
            
            # Log to W&B if online
            if self.online:
                wandb.log(
                    {
                        'Episode': self.ep_count,
                        'Metrics/Discomfort': self._log_fn(self.discomfort_h),
                        'Metrics/CO2_Emissions': self._log_fn(self.carbon_emissions_h),
                        'Metrics/Electricity_Consumption': self._log_fn(self.net_consumption_h),
                        'Metrics/EpRet': self._log_fn(self.ep_rewards),
                        'Metrics/EpLen': self._log_fn(self.ep_lengths),
                        'Metrics/EpCost': 0.0
                    }
                )
            
            # Print episode summary if verbose
            if self.verbose:
                print(
                    f"{'*'*30}\nEPISODE {self.ep_count}"                  +
                    f'\n- Discomfort:              {discomfort:.4f}'       +
                    f'\n- CO2 Emissions:           {carbon_emissions:.4f}' +
                    f'\n- Electricity Consumption: {net_consumption:.4f}'  +
                    f'\n- Reward:                  {self.current_ep_reward:.4f}' +
                    f'\n- Length:                  {self.current_ep_length}'
                )
        
        return obs, reward, terminated, truncated, info