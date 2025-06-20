from typing import Any, List, Mapping, Union, Tuple
import numpy as np
from citylearn.data import ZERO_DIVISION_PLACEHOLDER
from citylearn.reward_function import RewardFunction, ComfortReward

# Reward function that penalizes ramping in electricity consumption
class RampingCustom(RewardFunction):

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
        self.previous_obs = [] # stores previous observations to compute ramping

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        # Calculate ramping penalty or initialize with negative consumption
        if not self.previous_obs:
            for o in observations:
                r = min(-o['net_electricity_consumption'], 0)
                reward_list.append(r)
        else:
            # Penalize changes in consumption compared to previous timestep
            for o, p in zip(observations, self.previous_obs):
                r = -abs(o['net_electricity_consumption'] - p['net_electricity_consumption'])
                reward_list.append(r)
            self.previous_obs = observations

        # Sum rewards for central agent
        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward

# Reward function that penalizes blackouts and high consumption
class BlackoutCustom(RewardFunction):

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        for o, m in zip(observations, self.env_metadata['buildings']):
            blackout = o['power_outage']
            load = o['non_shiftable_load']
            dhw = o['dhw_demand']
            cooling = o['cooling_electricity_consumption']
            soc = o['electrical_storage_soc']
            comfort_penalty = o['average_unmet_cooling_setpoint_difference'] * 10

            # Heavier penalty during blackout, otherwise penalize consumption
            if blackout:
                reward = (- 10 * cooling - 5 * soc + 5 * load)
            else:
                reward = min(-o['net_electricity_consumption'],0)

            reward_list.append(reward)

        # Sum rewards for central agent
        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward

# Reward function promoting storage use and penalizing consumption
class RewardBaseCustom(RewardFunction):

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
        # self.previous_obs = [] # can be used for ramping if needed

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):
            e = o['net_electricity_consumption']
            s = o['solar_generation']
            cc = self.env_metadata['buildings'][m].get('cooling_storage', {}).get('attributes', {}).get('capacity', 0.0)
            hc = self.env_metadata['buildings'][m].get('heating_storage', {}).get('attributes', {}).get('capacity', 0.0)
            dc = self.env_metadata['buildings'][m].get('dhw_storage', {}).get('attributes', {}).get('capacity', 0.0)
            ec = self.env_metadata['buildings'][m].get('electrical_storage', {}).get('attributes', {}).get('capacity', 0.0)
            cs = o.get('cooling_storage_soc', 0.0)
            hs = o.get('heating_storage_soc', 0.0)
            ds = o.get('dhw_storage_soc', 0.0)
            es = o.get('electrical_storage_soc', 0.0)

            solar_reward = 0

            # Incentivize low net consumption when storage is available
            solar_reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_PLACEHOLDER else 0.0
            solar_reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_PLACEHOLDER else 0.0
            solar_reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_PLACEHOLDER else 0.0
            solar_reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_PLACEHOLDER else 0.0

            reward_list.append(solar_reward)

        # Sum rewards for central agent
        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward
