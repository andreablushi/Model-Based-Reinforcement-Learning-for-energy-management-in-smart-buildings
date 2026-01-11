# Our agents
import json
from typing import Any, Dict, List, OrderedDict

# CityLearn utils
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.base import Agent

# Utils
import os
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from agents.model_based.mbrl.util.kpi_utils import get_kpis
from citylearn.data import DataSet
from utils import *
import argparse
import yaml
import seaborn as sns
import numpy as np
from abc import ABC

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

class Config(ABC):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Configurations for Constrained RL on CityLearn")
        self._add_args()

    def _add_args(self):
        # Seed
        self.parser.add_argument('--seed', type=int, default=1, help="Experiment seed")

        # CityLearn dataset
        self.parser.add_argument('--data', type=str, default='citylearn_challenge_2023_phase_1', help="CityLearn dataset")

class TrainConfig(Config):
    def __init__(self):
        super().__init__()
        self._add_train_args()
        self.args = self.parser.parse_args()

    def _add_train_args(self):
        # Train args
        self.parser.add_argument('--device', type=str, default='cpu', help="CUDA device for training")
        self.parser.add_argument('--algo', type=str, default='PPO', help="RL algorithm to use")
        self.parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes to rollout")

        # CityLearn config
        self.parser.add_argument('--frac', type=float, default=1.0, help="Fraction of days in the dataset to use for training")
        self.parser.add_argument('--render', action='store_true', help="Flag for using `CityLearnEnv.render()`")
        self.parser.add_argument('--custom', action='store_true', help="Flag for CityLearn dataset customization")

        # Logging
        self.parser.add_argument('--name', type=str, nargs='?', help="Experiment name (used for directory)")
        self.parser.add_argument('--wandb', action='store_true', help="Flag for logging on wandb")
        self.parser.add_argument('--entity', type=str, default='universitaverona', help="Wandb entity")
        self.parser.add_argument('--project', type=str, default='citylearn_omnisafe', help="Wandb project name")
        self.parser.add_argument('--tag', type=str, default='comfort_reward', help="Wandb tag")

    def save_yaml(self, dir: str):
        with open(f'{dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)

class EvalConfig(Config):
    def __init__(self):
        super().__init__()
        self._add_eval_args()
        self.args = self.parser.parse_args()

    def _add_eval_args(self):
        # Eval args
        self.parser.add_argument('--exp_dir', type=str, default='./experiments/PPO_seed1_04-11-25_14:46:25', help="Path to the experiment of the RL agent to evaluate")
        self.parser.add_argument('--test', action='store_true', help="Evaluation mode")

        # CityLearn config
        self.parser.add_argument('--building', type=int, default=2, help="Whether to evaluate on the same building of training")

class ComfortRBC(OptimizedRBC):
    """
    Rule-based Control designed to overwrite controls scheduled by :py:class:`citylearn.agents.rbc.OptimizedRBC` 
    in order to tackle temperature discomfort.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment to perform control on.
    band: float
        Comfort band to try to satisfy. 

    TODO
    ---------- 
    Understand how to manage storages and devices with respect to them
    """
    def __init__(self, env: CityLearnEnv, band: float=None, **kwargs):        
        # Init OptimizedRBC
        super().__init__(env, **kwargs)

        # Sanity check
        self._check(env)

        # Comfort band (+/-) to satisfy
        self.comfort_band = band if band is not None else env.buildings[0].comfort_band[0] 

    def predict(self, observations: List[List[float]], deterministic: bool=None) -> List[List[float]]:        
        # Predict actions based on hour scheduling
        scheduled_acions = super().predict(observations, deterministic)

        actions = []
        for i, o in enumerate(observations):
            action = scheduled_acions[i]

            # Available spaces
            available_obs = self.observation_names[i]
            available_act = self.action_names[i]

            # Temperatures
            if 'indoor_dry_bulb_temperature' in available_obs:
                indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            else:
                indoor_temp = None

            if 'outdoor_dry_bulb_temperature' in available_obs:
                outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature')]
            else:
                outdoor_temp = None

            if 'indoor_dry_bulb_temperature_cooling_set_point' in available_obs:
                cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]
            else:
                cooling_setpoint = None

            if 'indoor_dry_bulb_temperature_cooling_delta' in available_obs:
                cooling_delta = o[available_obs.index('indoor_dry_bulb_temperature_cooling_delta')]
            else:
                cooling_delta = None

            if 'indoor_dry_bulb_temperature_heating_set_point' in available_obs:
                heating_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_heating_set_point')]
            else:
                heating_setpoint = None

            if 'indoor_dry_bulb_temperature_heating_delta' in available_obs:
                heating_delta = o[available_obs.index('indoor_dry_bulb_temperature_heating_delta')]
            else:
                heating_delta = None

            # Stoarges SoC
            if 'electrical_storage_soc' in available_obs:
                electrical_soc = o[available_obs.index('electrical_storage_soc')]
            else:
                electrical_soc = -1

            if 'cooling_storage_soc' in available_obs:
                cooling_soc = o[available_obs.index('cooling_storage_soc')]
            else:
                cooling_soc = -1

            if 'heating_storage_soc' in available_obs:
                heating_soc = o[available_obs.index('heating_storage_soc')]
            else:
                heating_soc = -1

            # Manage cooling
            if 'cooling_device' in available_act:
                # Action indexes
                device_idx = available_act.index('cooling_device')
                if 'electircal_storage' in available_act:
                    ess_idx = available_act.index('electrical_storage')
                else:
                    ess_idx = None

                # Temperature difference
                hot_delta = cooling_delta if cooling_delta is not None else indoor_temp - cooling_setpoint
                if hot_delta > 0:
                    if hot_delta > self.comfort_band: # Too hot -> supply the cooling device                        
                        action[device_idx] = 0.8
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/2)
                    else:
                        action[device_idx] = 0.2 # Hot within the band
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/3)
                else:
                    if indoor_temp is not None and outdoor_temp is not None:
                        temp_delta = outdoor_temp - indoor_temp # Outdoor temperature affects indoor temperature                       
                        action[device_idx] = 0.3 if temp_delta > 0 else 0.0

                    else:
                        action[device_idx] = 0.0
                        if ess_idx is not None: 
                            action[ess_idx] = action[ess_idx]/2 if action[ess_idx] < 0 else action[ess_idx]

            # Manage heating
            if 'heating_device' in available_act:
                # Action indexes
                device_idx = available_act.index('heating_device')
                if 'electrical_storage' in available_act:
                    ess_idx = available_act.index('electrical_storage')
                else:
                    ess_idx = None

                # Temperature difference
                cold_delta = heating_delta if heating_delta is not None else indoor_temp - heating_setpoint
                if cold_delta < 0:
                    if cold_delta < -self.comfort_band:
                        action[device_idx] = 0.8 # Too cold -> supply the heating device
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/2)
                    else:
                        action[device_idx] = 0.2 # Cold within the band
                        if electrical_soc > 0.1 and ess_idx is not None:
                            action[ess_idx] =  min(action[ess_idx], -electrical_soc/3)
                else:
                    if indoor_temp is not None and outdoor_temp is not None:
                        temp_delta = outdoor_temp - indoor_temp # Outdoor temperature affects indoor temperature
                        action[device_idx] = 0.3 if temp_delta < 0 else 0.0
                    else:
                        action[device_idx] = 0.0
                        if ess_idx is not None: 
                            action[ess_idx] = action[ess_idx]/2 if action[ess_idx] < 0 else action[ess_idx]

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions
    
    def _check(self, env: CityLearnEnv):
        if 'indoor_dry_bulb_temperature' in env.observation_names[0]:
            if 'indoor_dry_bulb_temperature_cooling_set_point' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_set_point' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_cooling_delta' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_delta' not in env.observation_names[0]:
                raise RuntimeError(
                    '`indoor_dry_bulb_temperature` is available, but no `indoor_dry_bulb_temperature_*_set_point` ' +
                    'or  `indoor_dry_bulb_temperature_*_delta` is available.'
                )
        else:
            if 'indoor_dry_bulb_temperature_cooling_delta' not in env.observation_names[0] \
                and 'indoor_dry_bulb_temperature_heating_delta' not in env.observation_names[0]:
                raise RuntimeError('No `indoor_dry_bulb_temperature_*_delta` is available.')

class AdvancedRBC(Agent):
    """
    Advanced Rule-Based Controller (RBC) Agent with comfort band consideration.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment to perform control on.
    band: float
        Comfort band to try to satisfy. 

    """
    def __init__(self, env: CityLearnEnv, band: float=2.0, **kwargs):

        # Init OptimizedRBC
        super().__init__(env, **kwargs)

        # Comfort band (+/-) to satisfy
        self.comfort_band = band 

    def predict(self, observations: List[List[float]], deterministic: bool = True) -> List[List[float]]:        
            
        actions = []
        for i, o in enumerate(observations):

            # Available spaces
            available_obs = self.observation_names[i]
            available_act = self.action_names[i]
            action = [0.0 for _ in range(len(available_act))]

            # Indoor temperature and setpoints
            indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]

            # Outdoor temperature
            outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature')]
            predicted_outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature_predicted_1')]

            # Cooling demand
            cooling_demand = o[available_obs.index('cooling_demand')]

            # Electricity pricing
            elec_price = o[available_obs.index('electricity_pricing')]

            # Emission
            carbon_int = o[available_obs.index('carbon_intensity')]

            # Solar generation
            solar_gen = o[available_obs.index('solar_generation')]

            # Occupants presence
            occupants_present = o[available_obs.index('occupant_count')]    

            # Hours of the day
            hour = o[available_obs.index('hour')]

            # Electrical storage state of charge
            electrical_storage_soc = o[available_obs.index('electrical_storage_soc')]

            # DHW storage state of charge
            dhw_storage_soc = o[available_obs.index('dhw_storage_soc')]

            # DHW demand
            dhw_demand = o[available_obs.index('dhw_demand')]

            if 'cooling_device' in available_act:
                # Peak hours
                if 12 <= hour <= 17:
                    if indoor_temp > cooling_setpoint + self.comfort_band:
                        if occupants_present == 0:
                            action[available_act.index('cooling_device')] = 0.0
                        else:
                            # Carbon emission evaluation
                            if carbon_int < 0.40:
                                # Low electricity price
                                if elec_price <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.8
                                    else:
                                        action[available_act.index('cooling_device')] = 1.0
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.8
                                # Low electricity price + high solar generation
                                elif elec_price <= 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.85
                                    else:
                                        action[available_act.index('cooling_device')] = 0.1
                                # High electricity price + high solar generation
                                elif elec_price > 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    else:
                                        action[available_act.index('cooling_device')] = 0.9
                                # Default action
                                    action[available_act.index('cooling_device')] = 0.66
                            # High carbon emission
                            else:
                                # Low electricity price
                                if elec_price <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    else:
                                        action[available_act.index('cooling_device')] = 0.9
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    else:
                                        action[available_act.index('cooling_device')] = 0.7
                                # Low electricity price + high solar generation
                                elif elec_price <= 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.8
                                    else:
                                        action[available_act.index('cooling_device')] = 1.0
                                # High electricity price + high solar generation
                                elif elec_price > 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # Default action
                                action[available_act.index('cooling_device')] = 0.55
                    
                    # Cooling demand evaluation
                    else:
                        if 0.7 <= cooling_demand <= 1.0:
                            action[available_act.index('cooling_device')] = 0.66
                        elif 0.3 <= cooling_demand < 0.7:
                            action[available_act.index('cooling_device')] = 0.4
                        else:
                            action[available_act.index('cooling_device')] = 0.2

                # Off-peak hours
                else:
                    # Indoor temperature above setpoint + comfort band
                    if indoor_temp > cooling_setpoint + self.comfort_band:
                        # Carbon emission evaluation
                        if carbon_int < 0.40:
                            # Low electricity price
                            if elec_price <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.6
                                else:
                                    action[available_act.index('cooling_device')] = 0.8
                            # High electricity price
                            elif elec_price > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.25
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # Low electricity price + high solar generation
                            if elec_price <= 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.7
                                else:
                                    action[available_act.index('cooling_device')] = 0.9
                            # High electricity price + high solar generation
                            if elec_price > 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                else:
                                    action[available_act.index('cooling_device')] = 0.7
                            # Default action
                            action[available_act.index('cooling_device')] = 0.5
                        # High carbon emission
                        else:
                            # Low electricity price
                            if elec_price <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                else:
                                    action[available_act.index('cooling_device')] = 0.7
                            # High electricity price
                            elif elec_price > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                else:
                                    action[available_act.index('cooling_device')] = 0.5
                            # Low electricity price + high solar generation
                            if elec_price <= 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.6
                                else:
                                    action[available_act.index('cooling_device')] = 0.8
                            # High electricity price + high solar generation
                            if elec_price > 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.25
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # Default action
                            action[available_act.index('cooling_device')] = 0.44
                    
                    # Cooling demand evaluation
                    else:
                        if 0.7 <= cooling_demand <= 1.0:
                            action[available_act.index('cooling_device')] = 0.4
                        elif 0.3 <= cooling_demand < 0.7:
                            action[available_act.index('cooling_device')] = 0.2
                        else:
                            action[available_act.index('cooling_device')] = 0.1

            if 'electrical_storage' in available_act:
                if electrical_storage_soc == 1.0:
                    # Peak hours -> discharge
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        # High solar generation
                        if solar_gen > 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = -0.66
                            else:
                                action[available_act.index('electrical_storage')] = -0.85
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = -0.45
                            else:
                                action[available_act.index('electrical_storage')] = -0.6
                        else:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = -0.3
                            else:
                                action[available_act.index('electrical_storage')] = -0.4
                    # Off-peak hours -> battery at max capacity so no action
                    else:
                        action[available_act.index('electrical_storage')] = 0.0

                elif 0.5 <= electrical_storage_soc < 1.0:
                    # Peak hours -> discharge
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        # High solar generation
                        if solar_gen > 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = -0.5
                            else:
                                action[available_act.index('electrical_storage')] = -0.6
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = -0.35
                            else:
                                action[available_act.index('electrical_storage')] = -0.45
                        else:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = -0.15
                            else:
                                action[available_act.index('electrical_storage')] = -0.3
                    # Off-peak hours
                    else:
                        # High solar generation
                        if solar_gen > 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.5
                            else:
                                action[available_act.index('electrical_storage')] = 0.35
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.4
                            else:
                                action[available_act.index('electrical_storage')] = 0.25
                        else:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.3
                            else:
                                action[available_act.index('electrical_storage')] = 0.15
                
                elif 0.2 <= electrical_storage_soc < 0.5:
                    # Peak hours -> low battery capacity so minimal discharge
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        action[available_act.index('electrical_storage')] = -0.1
                    # Off-peak hours -> charge
                    else:
                        # High solar generation
                        if solar_gen > 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.7
                            else:
                                action[available_act.index('electrical_storage')] = 0.55
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.65
                            else:
                                action[available_act.index('electrical_storage')] = 0.4
                        else:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.5
                            else:
                                action[available_act.index('electrical_storage')] = 0.3
                
                # Very low state of charge -> charge
                else:
                    # Peak hours -> no action
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        action[available_act.index('electrical_storage')] = 0.0
                    # Off-peak hours -> charge
                    else:
                        # High solar generation
                        if solar_gen > 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.9
                            else:
                                action[available_act.index('electrical_storage')] = 0.6
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.7
                            else:
                                action[available_act.index('electrical_storage')] = 0.45
                        else:
                            if elec_price < 0.03:
                                action[available_act.index('electrical_storage')] = 0.6
                            else:
                                action[available_act.index('electrical_storage')] = 0.4

            if 'dhw_storage' in available_act:
                if 0.7 <= dhw_storage_soc <= 1.0:
                    # Peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        # High demand or high electricity price
                        if dhw_demand >= 0.6 or elec_price > 0.03:
                            action[available_act.index('dhw_storage')] = -0.6
                        # Low demand
                        elif dhw_demand < 0.3:
                            action[available_act.index('dhw_storage')] = -0.2
                        # Default action
                        else:
                            action[available_act.index('dhw_storage')] = -0.35
                    # Off-peak hours with possible high solar generation
                    elif 10 <= hour <= 16:
                        # High solar generation and low electricity price
                        if solar_gen > 0.6 and elec_price < 0.03:
                            action[available_act.index('dhw_storage')] = 0.4
                        # Medium solar generation
                        elif 0.2 <= solar_gen <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.25
                        # Low solar generation -> default action
                        else:
                            action[available_act.index('dhw_storage')] = 0.0
                    # Night hours -> high dhw storage state of charge so no action
                    else:
                        action[available_act.index('dhw_storage')] = 0.0
                
                elif 0.4 <= dhw_storage_soc < 0.7:
                    # Peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        # High demand or high electricity price
                        if dhw_demand >= 0.6 or elec_price > 0.03:
                            action[available_act.index('dhw_storage')] = -0.4
                        # Low demand
                        elif dhw_demand < 0.3:
                            action[available_act.index('dhw_storage')] = -0.15
                        # Default action
                        else:
                            action[available_act.index('dhw_storage')] = -0.25
                    # Off-peak hours with possible high solar generation
                    elif 10 <= hour <= 16:
                        # High solar generation and low electricity price
                        if solar_gen > 0.6 and elec_price < 0.03:
                            action[available_act.index('dhw_storage')] = 0.65
                        # Medium solar generation
                        elif 0.2 <= solar_gen <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.35
                        # Low solar generation -> default action
                        else:
                            action[available_act.index('dhw_storage')] = 0.15
                    # Night hours -> high dhw storage state of charge so no action
                    else:
                        action[available_act.index('dhw_storage')] = 0.5

                # Very low state of charge
                else:
                    # Peak hours -> minimal discharge
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        action[available_act.index('dhw_storage')] = -0.01
                    # Off-peak hours with possible high solar generation
                    elif 10 <= hour <= 16:
                        # High solar generation and low electricity price
                        if solar_gen > 0.6 and elec_price < 0.03:
                            action[available_act.index('dhw_storage')] = 0.8
                        # Medium solar generation
                        elif 0.2 <= solar_gen <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.6
                        # Low solar generation -> default action
                        else:
                            action[available_act.index('dhw_storage')] = 0.4
                    # Night hours -> low dhw storage state of charge so charge
                    else:
                        action[available_act.index('dhw_storage')] = 0.7

            # Debugging log for actions values and observations values -> Can be modified as needed
            # Actions value logging should be implemented in a more structured way for UserRBC
            debug_action_dict = {}
            if 'cooling_device' in available_act:
                debug_action_dict['cooling_device'] = action[available_act.index('cooling_device')]
            if 'dhw_storage' in available_act:
                debug_action_dict['dhw_storage'] = action[available_act.index('dhw_storage')]
            if 'electrical_storage' in available_act:
                debug_action_dict['electrical_storage'] = action[available_act.index('electrical_storage')]

            # print(f"[DEBUG] Hour (from obs): {hour:.0f}, Actions: {debug_action_dict}")

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions


def plot_temperature(res_1, suffix=''):
    temp_1 = res_1['env_h']['temperature']
    indoor_temps = temp_1['indoor_temperature']
    indoor_setpoints = temp_1['indoor_temperature_set_point']
    outdoor_temps = temp_1['outdoor_temperature']
    comfort_band = temp_1['comfort_band']

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 5)

    # Ensure arrays match in length
    n = len(indoor_temps)

    # --- Create figure and subplots ---
    fig, axs = plt.subplots(1, 1, figsize=(30, 10))
    # 0️⃣ Temperature profiles
    axs.plot(range(n), indoor_temps, label='Indoor Temperature', color=palette[0], lw=2)
    # Set point comfort band
    axs.fill_between(
        range(n),
        indoor_setpoints + comfort_band,
        indoor_setpoints - comfort_band,
        color='g',
        alpha=0.1,
        label='Comfort band',
    )
    axs.plot(range(n), outdoor_temps, label='Outdoor Temperature', color=palette[1], lw=2)
    axs.set_ylabel('Temperature [°C]')
    axs.set_title('Temperature Profiles')
    axs.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f'temperature_profile_{suffix}.png', dpi=300, bbox_inches='tight')


def compare_temperature(args, res_1, res_2, algo_names=[]):
    # RBC temperature
    temp_1 = res_1['env_h']['temperature']
    temp_2 = res_2['env_h']['temperature']

    
    fig = plt.figure(figsize=[30,10])
    fig.suptitle('Temperature management')
    ax = fig.add_subplot(1,1,1)

    # Set point comfort band
    ax.fill_between(
        range(res_1['env_h']['time_steps']),
        temp_1['indoor_temperature_set_point'] + temp_1['comfort_band'],
        temp_1['indoor_temperature_set_point'] - temp_1['comfort_band'],
        color='g',
        alpha=0.1,
        label='Comfort band',
    )

    # Control temperature
    ax.plot(temp_1['indoor_temperature'], label=algo_names[0] if algo_names else 'RBC', linewidth=2.0)
    ax.plot(temp_2['indoor_temperature'], label=algo_names[1] if algo_names else 'RL', linewidth=2.0)

    ax.set_ylabel('Temperature (°C)')
    plt.legend()

    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/temperature.png', format='png')

def compare_battery(args, res_1, res_2, algo_names=[]):
    # Set figure
    fig = plt.figure(figsize=[20,15])
    fig.suptitle('Electrical storage history')

    # Control 1
    battery_1 = res_1['env_h']['battery']
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title(algo_names[0] if algo_names else 'Rule-Based Control')
    # Charge rate
    ax1.bar(range(res_1['env_h']['time_steps']-1), battery_1['discharge'], color='xkcd:soft blue')
    ax1.set_ylabel('(Dis)Charge (kW/h)')
    ax1.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    ax2 = ax1.twinx()
    ax2.plot(battery_1['soc'], linewidth=2.0, c='xkcd:orange')
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(ymin=-0.05, ymax=1.05)
    ax2.yaxis.label.set_color('xkcd:orange')

    # Control 2
    battery_2 = res_2['env_h']['battery']
    ax3 = fig.add_subplot(2,1,2)
    ax3.set_title(algo_names[1] if algo_names else 'Reinforcement Learning Control')
    # Charge rate
    ax3.bar(range(res_2['env_h']['time_steps']-1), battery_2['discharge'], color='xkcd:soft blue')
    ax3.set_ylabel('(Dis)Charge (kW/h)')
    ax3.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    ax4 = ax3.twinx()
    ax4.plot(battery_2['soc'], linewidth=2.0, c='xkcd:orange')
    ax4.set_ylabel('SoC (%)')
    ax4.set_ylim(ymin=-0.05, ymax=1.05)
    ax4.yaxis.label.set_color('xkcd:orange')

    mode = 'test' if args.test else 'eval'
    os.makedirs(f'{args.exp_dir}/{mode}_figs', exist_ok=True)
    fig.savefig(f'{args.exp_dir}/{mode}_figs/battery.png', format='png')

def plot_energy(
    res,
    suffix=''
):
    cooling_device_consumption=res['env_h']['cooling_device']['consumption']
    dhw_device_consumption=res['env_h']['dhw']['consumption']
    non_shiftable_load=res['env_h']['non_shiftable_load']
    battery_charge=res['env_h']['battery']['consumption']
    pv_generation=res['env_h']['solar_generation']
    battery_action=res['env_h']['battery']['discharge']
    battery_soc=res['env_h']['battery']['soc']
    net_load=res['env_h']['net_electricity_consumption']

    time = range(len(cooling_device_consumption))

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 6)

    # --- Prepare data ---
    pv_generation = -1 * pv_generation  # Flip sign for plotting

    # --- Derived quantities ---
    building_demand = (
        cooling_device_consumption
        + dhw_device_consumption
        + non_shiftable_load
        + battery_charge
    )

    battery_power = battery_action
    label_action = "(Dis)Charge [kW/h]"

    # --- Figure setup ---
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    plt.subplots_adjust(hspace=0.35, right=0.85)

    # 0️⃣ Building Consumption (stacked)
    components = np.vstack([
        cooling_device_consumption,
        dhw_device_consumption,
        non_shiftable_load,
        battery_charge
    ])
    labels = ['Cooling', 'DHW', 'Non-shiftable', 'Battery (Charging)']
    colors = palette[:len(labels)]

    axs[0].stackplot(time, components, labels=labels, colors=colors, alpha=0.9)
    axs[0].plot(time, building_demand, color='black', lw=2, label='Total')
    axs[0].set_ylabel('Power [kW]')
    axs[0].set_title('Building Electricity Consumption Components')

    # Legend on the side (slightly higher)
    legend = axs[0].legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # 1️⃣ Building Demand, PV, Net Load
    axs[1].plot(time, building_demand, label='Building Demand', color='gray', lw=1.8)
    axs[1].fill_between(time, 0, pv_generation, color=palette[2], alpha=0.3, label='PV Generation')
    axs[1].plot(time, net_load, label='Net Load', color='black', lw=2)
    axs[1].set_ylabel('Power [kW]')
    axs[1].set_title('Building Demand, PV Generation, and Net Load')
    axs[1].legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # 2️⃣ Battery Action / Power
    axs[2].axhline(0, color='black', lw=0.8)
    sns.lineplot(x=time, y=battery_power, ax=axs[2], color=palette[0], lw=1.8)
    axs[2].set_ylabel(label_action)
    axs[2].set_title('Battery Control Signal (Action)')
    axs[2].set_ylim(-1.1 * np.max(np.abs(battery_power)), 1.1 * np.max(np.abs(battery_power)))

    # 3️⃣ Battery SoC
    sns.lineplot(x=time, y=battery_soc, ax=axs[3], color=palette[4], lw=2)
    axs[3].set_ylabel('State of Charge [%]')
    axs[3].set_xlabel('Time')
    axs[3].set_title('Battery State of Charge (SoC)')

    # Add explanatory note below the first subplot (figure-level annotation)
    fig.text(
        0.80, 0.62,  # position relative to the figure (x, y)
        "Net Load meaning:\n"
        "   • Net Load > 0 → Import from grid\n"
        "   • Net Load < 0 → Export to grid",
        ha='left',
        va='top',
        fontsize=11,
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            edgecolor='gray',
            boxstyle='round,pad=0.4'
        )
    )

    plt.tight_layout()  # leave extra space on right
    plt.savefig(
        f'energy_profile_{suffix}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def compare_kpis(args, res_1, res_2, algo_names=[]):
    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 5)

    kpis_1 = res_1['kpis']
    kpis_2 = res_2['kpis']

    # Create a DataFrame for the KPIs
    kpi_names = list(kpis_1.keys())
    values_1 = [kpis_1[kpi] for kpi in kpi_names]
    values_2 = [kpis_2[kpi] for kpi in kpi_names]

    kpi_df = pd.DataFrame({
        'KPI': kpi_names,
        'Res 1': values_1,
        'Res 2': values_2
    })

    # Set up the horizontal bar plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(kpi_names))

    # Create horizontal bars for both results
    bar1 = plt.barh(index, kpi_df['Res 1'], bar_width, label=algo_names[0], color=palette[0])
    bar2 = plt.barh(index + bar_width, kpi_df['Res 2'], bar_width, label=algo_names[1], color=palette[1])

    # Add labels and title
    plt.ylabel('KPIs')
    plt.xlabel('Values')
    plt.title(f'Comparison of KPIs between {algo_names[0]} and {algo_names[1]}')
    plt.yticks(index + bar_width / 2, kpi_names)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f'kpi_comparison.png', dpi=300, bbox_inches='tight')

def evaluate(args, agent_type: str, schema: dict, seed: int=None):
    # Create CityLearn environment
    env = CityLearnEnv(
        schema=schema, 
        central_agent=True,
    )
    # Agent
    if agent_type == 'comfort_rbc':
        agent = ComfortRBC(env)
    elif agent_type == 'advanced_rbc':
        agent = AdvancedRBC(env)
    elif agent_type == 'custom_rbc':
        # agent = CustomRBC(env)
        agent = ComfortRBC(env)
    else:
        raise RuntimeError(f'Unknown agent type {agent_type}. Must be either `rbc` or `rl`.')
    
    # Episodic return
    results = {}
    ep_reward = 0.0

    # Step through the environment
    obs, _ = env.reset(seed=args.seed)
    while not env.terminated:
        action = agent.predict(obs)        
        obs, reward, _, _, _ = env.step(action)
        ep_reward += reward[0]

    # Get KPIs
    kpis = get_kpis(env=env)

    # Console log
    print(
        f"{'*'*30}\n CONTROL RESULTS ({agent_type}{f' | seed={seed}' if seed is not None else ''})" +
        f'\n- Reward: {ep_reward}'
    )

    for kpi, value in kpis.items():
        print(f'- {kpi}: {value:.2f}')

    print(f"{'*'*30}")

    # Populate results dict
    results['kpis'] = kpis
    results['env_h'] = {
        'time_steps': env.time_steps,
        'temperature': {
            'indoor_temperature': env.buildings[0].indoor_dry_bulb_temperature,
            'indoor_temperature_set_point': env.buildings[0].indoor_dry_bulb_temperature_cooling_set_point,
            'outdoor_temperature': env.buildings[0].weather.outdoor_dry_bulb_temperature ,
            'comfort_band': env.buildings[0].comfort_band
        },
        'battery': {
            'soc': env.buildings[0].electrical_storage.soc[:-1],
            'discharge': env.buildings[0].electrical_storage.energy_balance[:-1],
            'consumption': env.buildings[0].electrical_storage.electricity_consumption[:-1]
        },
        'dhw': {
            'soc': env.buildings[0].dhw_storage.soc[:-1],
            'demand': env.buildings[0].dhw_demand[:-1],
            'consumption': env.buildings[0].dhw_electricity_consumption[:-1]
        },
        'cooling_device': {
            'consumption': env.buildings[0].cooling_device.electricity_consumption[:-1]
        },
        'net_electricity_consumption': env.buildings[0].net_electricity_consumption[:-1],
        'solar_generation': env.buildings[0].solar_generation[:-1],
        'non_shiftable_load': env.buildings[0].non_shiftable_load[:-1],
        'electricity_pricing': env.buildings[0].pricing.electricity_pricing[:-1],
    }

    return results


if __name__ == '__main__':
    config = EvalConfig()
    args = config.args 

    os.makedirs(args.exp_dir, exist_ok=True)

    # Get schema from CityLearn dataset
    schema_obj = CityLearnSchema()
    schema_obj.load(dataset=args.data)

    # Modify schema for testing on a different building
    schema_obj.set_active(key='buildings', items=[f'Building_{args.building}'])

    # Evaluate Rule-based Control agent
    res_comfort_rbc = evaluate(args, 'comfort_rbc', schema_obj.schema)
    res_advanced_rbc = evaluate(args, 'advanced_rbc', schema_obj.schema)

    # Compare results
    compare_temperature(args, res_comfort_rbc, res_advanced_rbc, algo_names=['Comfort RBC', 'Advanced RBC'])
    compare_battery(args, res_comfort_rbc, res_advanced_rbc, algo_names=['Comfort RBC', 'Advanced RBC'])
    compare_kpis(args, res_comfort_rbc, res_advanced_rbc, algo_names=['Comfort RBC', 'Advanced RBC'])
    # Plot energy profile for Comfort RBC
    plot_energy(
        res= res_comfort_rbc,
        suffix='comfort_rbc'
    )
    # Plot energy profile for Advanced RBC
    plot_energy(
        res_advanced_rbc,
        suffix='advanced_rbc'
    )

    plot_temperature(
        res_1=res_comfort_rbc,
        suffix='comfort_rbc'
    )

    plot_temperature(
        res_1=res_advanced_rbc,
        suffix='advanced_rbc'
    )