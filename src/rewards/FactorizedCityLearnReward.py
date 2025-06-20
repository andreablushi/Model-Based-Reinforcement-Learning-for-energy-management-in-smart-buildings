from typing import Any, List, Mapping, Union, Tuple
import numpy as np
import math
from citylearn.data import ZERO_DIVISION_PLACEHOLDER
from citylearn.reward_function import RewardFunction

"""
Di seguito ho inserito una copia dell'implementazione di SolarPenaltyReward e SolarPenaltyAndComfortReward, con un mia correzione in quanto quelle della libreria originale non funzionano.
"""
    
class SolarPenaltyReward(RewardFunction):
    """The reward is designed to minimize electricity consumption and maximize solar generation to charge energy storage systems.

    The reward is calculated for each building, i and summed to provide the agent with a reward that is representative of all the
    building or buildings (in centralized case)it controls. It encourages net-zero energy use by penalizing grid load satisfaction 
    when there is energy in the energy storage systems as well as penalizing net export when the energy storage systems are not
    fully charged through the penalty term. There is neither penalty nor reward when the energy storage systems are fully charged
    during net export to the grid. Whereas, when the energy storage systems are charged to capacity and there is net import from the 
    grid the penalty is maximized.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

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
            reward = 0.0
            # reward += s * es * 4 #utile ad incentivare l'utilizzo della batteria, da usare con cura
            reward += -(1.0 + np.sign(e)*cs)*abs(e) if cc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*hs)*abs(e) if hc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*ds)*abs(e) if dc > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward += -(1.0 + np.sign(e)*es)*abs(e) if ec > ZERO_DIVISION_PLACEHOLDER else 0.0
            reward_list.append(reward)

        return reward_list

    
class ComfortReward(RewardFunction):
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
                
                elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                    reward = 0.0 if heating else -delta

                elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                    reward = -delta if heating else 0.0

                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(delta**exponent)

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
        return reward_list

    
class FactorizedSolarPenaltyAndComfortReward(RewardFunction):
    """Addition of :py:class:`citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward`.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-). If not provided, the comfort band time series defined in the
        building file, or the default time series value of 2.0 is used.
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 3.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    coefficients: Tuple, default = (1.0, 1.0)
        Coefficents for `citylearn.reward_function.SolarPenaltyReward` and :py:class:`citylearn.reward_function.ComfortReward` values respectively.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None, coefficients: Tuple = None):
        self.__functions: List[RewardFunction] = [
            SolarPenaltyReward(env_metadata),
            ComfortReward(env_metadata, band=band, lower_exponent=lower_exponent, higher_exponent=higher_exponent),
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        RewardFunction.env_metadata.fset(self, env_metadata)

        for f in self.__functions:
            f.env_metadata = self.env_metadata
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward = np.array([f.calculate(observations) for f in self.__functions], dtype='float32')
        reward = reward*np.reshape(self.coefficients, (len(self.coefficients), 1))
        reward = reward.sum(axis=0).tolist()
        return [reward]

