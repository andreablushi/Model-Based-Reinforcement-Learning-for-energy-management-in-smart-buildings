from typing import Any, List, Mapping, Union
from citylearn.reward_function import ComfortReward as ComfortRewardFunction

"""
Reward of first in leaderboard CityLearn Post-Challenge 2023 : dhchen
"""

class ComfortandConsumptionReductionReward(ComfortRewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any]):
        self.multiplier = 3.0
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        total_building_consumption_minus1 = 0 
        r = 0 
        s = 0 

        for o in observations:
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            hvac_mode = o['hvac_mode']
            set_point = o['indoor_dry_bulb_temperature_cooling_set_point'] if hvac_mode == 1 else o['indoor_dry_bulb_temperature_heating_set_point']
            delta = indoor_dry_bulb_temperature - set_point

            if delta < 0.0:
                delta *= self.multiplier
            else:
                pass
            
            
            #Ramping
            total_building_consumption = o['net_electricity_consumption']
            if total_building_consumption_minus1 != 0: 
                r = total_building_consumption - total_building_consumption_minus1 
            total_building_consumption_minus1 = total_building_consumption
            
            '''
            cooling_demand = o['coooling_demand']
            dhw_demand = o['dhw_demand']
            cooling_electricity_consumption = o['cooling_electricity_consumption']
            dhw_electricity_consumption = o['dhw_electricity_consumption']
            '''
            # Carbon emissions 
            #carbon_emissions_rate = o['carbon_intensity']
            #g = max(0,total_building_consumption*carbon_emissions_rate)

            # Poorly done normalized unserved energy 
            outage = o['power_outage'] 
            non_shiftable_load = o['non_shiftable_load']

            if outage == 1: 
                s_expected = non_shiftable_load 
                s_served = total_building_consumption
                s = s_expected - s_served

            else: 
                pass

            reward = -0.3*abs(delta) + -0.1*(s) + -0.15*abs(r)#-0.05*(g-7000)
            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)/len(reward_list)]

        else:
            reward = reward_list

        return reward
