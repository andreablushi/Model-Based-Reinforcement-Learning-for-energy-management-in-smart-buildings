from typing import Any, List, Mapping, Union, Tuple
import numpy as np
from citylearn.data import ZERO_DIVISION_PLACEHOLDER
from citylearn.reward_function import RewardFunction
    
    
class ComfortCustom(RewardFunction):
    """Reward for occupant thermal comfort satisfaction.

    The reward is calculated as the negative difference between the setpoint and indoor dry-bulb temperature raised to some exponent
    if outside the comfort band. If within the comfort band, the reward is the negative difference when in cooling mode and temperature
    is below the setpoint or when in heating mode and temperature is above the setpoint. The reward is 0 if within the comfort band
    and above the setpoint in cooling mode or below the setpoint and in heating mode.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default: 2.0
        Setpoint comfort band (+/-). If not provided, the comfort band time series defined in the
        building file, or the default time series value of 2.0 is used.
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        super().__init__(env_metadata)
        self.band = band
        self.lower_exponent = lower_exponent
        self.higher_exponent = higher_exponent

    @property
    def band(self) -> float:
        return self.__band
    
    @property
    def lower_exponent(self) -> float:
        return self.__lower_exponent
    
    @property
    def higher_exponent(self) -> float:
        return self.__higher_exponent
    
    @band.setter
    def band(self, band: float):
        self.__band = band

    @lower_exponent.setter
    def lower_exponent(self, lower_exponent: float):
        self.__lower_exponent = 2.0 if lower_exponent is None else lower_exponent

    @higher_exponent.setter
    def higher_exponent(self, higher_exponent: float):
        self.__higher_exponent = 2.0 if higher_exponent is None else higher_exponent

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o in observations:
            heating_demand = o.get('heating_demand', 0.0)
            cooling_demand = o.get('cooling_demand', 0.0)
            heating = heating_demand > cooling_demand
            hvac_mode = o['hvac_mode']
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']

            if hvac_mode in [1, 2]:
                set_point = o['indoor_dry_bulb_temperature_cooling_set_point'] if hvac_mode == 1 else o['indoor_dry_bulb_temperature_heating_set_point']
                band =  self.band if self.band is not None else o['comfort_band']
                lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - band
                upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + band
                delta = abs(indoor_dry_bulb_temperature - set_point)
                
                if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                    exponent = self.lower_exponent if hvac_mode == 2 else self.higher_exponent
                    reward = -(delta**exponent)
                    # Aggiunge penalità x3 se in raffreddamento ed uscita dal limite inferiore
                    if hvac_mode == 1:
                        reward *= 2
                
                elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                    reward = 0.0 if heating else -delta

                elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                    reward = -delta if heating else 0.0

                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(delta**exponent)
                    # Aggiunge penalità x3 se in riscaldamento ed uscita dal limite superiore
                    if hvac_mode == 2:
                        reward *= 2

            else:
                cooling_set_point = o['indoor_dry_bulb_temperature_cooling_set_point']
                heating_set_point = o['indoor_dry_bulb_temperature_heating_set_point']
                band =  self.band if self.band is not None else o['comfort_band']
                lower_bound_comfortable_indoor_dry_bulb_temperature = heating_set_point - band
                upper_bound_comfortable_indoor_dry_bulb_temperature = cooling_set_point + band
                cooling_delta = indoor_dry_bulb_temperature - cooling_set_point
                heating_delta = indoor_dry_bulb_temperature - heating_set_point

                if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                    exponent = self.higher_exponent if not heating else self.lower_exponent
                    reward = -(abs(heating_delta)**exponent)
                    

                elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < heating_set_point:
                    reward = -(abs(heating_delta))

                elif heating_set_point <= indoor_dry_bulb_temperature <= cooling_set_point:
                    reward = 0.0

                elif cooling_set_point < indoor_dry_bulb_temperature < upper_bound_comfortable_indoor_dry_bulb_temperature:
                    reward = -(abs(cooling_delta))

                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(abs(cooling_delta)**exponent)

            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward
