
from typing import Dict
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from citylearn.citylearn import CityLearnEnv, EvaluationCondition

def get_kpis(env: CityLearnEnv) -> Dict[str, float]:
    kpis = env.unwrapped.evaluate()

    # KPIs to retrieve
    kpi_names = {
        'cost_total': 'Cost ($/kWh)',
        'carbon_emissions_total': 'Emissions (kgC02e/kWh)',
        'daily_peak_average': 'Avg. daily peak (kWh)',
        'ramping_average': 'Ramping (kWh)',
        'daily_one_minus_load_factor_average': '1 - load factor',
        'discomfort_proportion': 'Discomfort (%)'
    }

    # Filter KPIs
    kpis = kpis[
        (kpis['level'] == 'district') &
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()
    kpis['cost_function'] = kpis['cost_function'].map(lambda x: kpi_names[x])

    kpis_dict = {}
    for _, kpi in kpis.iterrows():
        kpis_dict[kpi['cost_function']] = kpi['value']

    return kpis_dict

def plot_building_kpis(envs: dict[str, CityLearnEnv]) -> pd.DataFrame:
    """Plots electricity consumption, cost and carbon emissions
    at the building-level for different control agents in bar charts.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='building'].copy()
        kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
        kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names= kpis['kpi'].unique()
    column_count_limit = 3
    row_count = math.ceil(len(kpi_names)/column_count_limit)
    column_count = min(column_count_limit, len(kpi_names))
    building_count = len(kpis['name'].unique())
    env_count = len(envs)
    figsize = (3.0*column_count, 0.8*env_count*building_count*row_count)
    fig, _ = plt.subplots(
        row_count, column_count, figsize=figsize, sharey=True
    )

    for i, (ax, (k, k_data)) in enumerate(zip(fig.axes, kpis.groupby('kpi'))):
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(k)

        for j, _ in enumerate(envs):
            ax.bar_label(ax.containers[j], fmt='%.2f')

        if i == len(kpi_names) - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

        for s in ['right','top']:
            ax.spines[s].set_visible(False)

    return kpis



def plot_district_kpis(envs: dict[str, CityLearnEnv]) -> pd.DataFrame:
    """Plots electricity consumption, cost, carbon emissions,
    average daily peak, ramping and (1 - load factor) at the
    district-level for different control agents in a bar chart.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='district'].copy()
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    row_count = 1
    column_count = 1
    env_count = len(envs)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (6.0*column_count, 0.225*env_count*kpi_count*row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)
    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    for j, _ in enumerate(envs):
        ax.bar_label(ax.containers[j], fmt='%.2f')

    for s in ['right','top']:
        ax.spines[s].set_visible(False)

    ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0)

    return kpis

def evaluate_citylearn_challenge(env: CityLearnEnv, weights: dict[str, float]) -> dict[str, float]:
    evaluation = {
            'carbon_emissions_total': {'display_name': 'Carbon emissions', 'weight': 0.10},
            'discomfort_proportion': {'display_name': 'Unmet hours', 'weight': 0.30},
            'ramping_average': {'display_name': 'Ramping', 'weight': 0.075},
            'daily_one_minus_load_factor_average': {'display_name': 'Load factor', 'weight': 0.075},
            'daily_peak_average': {'display_name': 'Daily peak', 'weight': 0.075},
            'all_time_peak_average': {'display_name': 'All-time peak', 'weight': 0.075},
            'one_minus_thermal_resilience_proportion': {'display_name': 'Thermal resilience', 'weight': 0.15},
            'power_outage_normalized_unserved_energy_total': {'display_name': 'Unserved energy', 'weight': 0.15},
    }
    data = env.unwrapped.evaluate(
            control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
            baseline_condition=EvaluationCondition.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV,
            comfort_band=2.0,
    )

    data = data[data['level']=='district'].set_index('cost_function').to_dict('index')
    evaluation = {k: {**v, 'value': data[k]['value']} for k, v in evaluation.items()}

    score_comfort = evaluation['discomfort_proportion']['value']
    score_emissions = evaluation['carbon_emissions_total']['value']
    score_grid_control = (
        evaluation['ramping_average']['value'] +
        evaluation['daily_one_minus_load_factor_average']['value'] +
        evaluation['daily_peak_average']['value'] +
        evaluation['all_time_peak_average']['value']
    ) / 4.0
    score_resilience = (
        evaluation['one_minus_thermal_resilience_proportion']['value'] +
        evaluation['power_outage_normalized_unserved_energy_total']['value']
    ) / 2.0

    evaluation['score_comfort'] = {
        'display_name': 'Comfort score',
        'weight': weights['comfort'],
        'value': score_comfort
    }
    evaluation['score_emissions'] = {
        'display_name': 'Emissions score',
        'weight': weights['emissions'],
        'value': score_emissions
    }
    evaluation['score_grid_control'] = {
        'display_name': 'Grid control score',
        'weight': weights['grid_control'],
        'value': score_grid_control
    }
    evaluation['score_resilience'] = {
        'display_name': 'Resilience score',
        'weight': weights['resilience'],
        'value': score_resilience
    }
    weighted_resilience = weights['resilience'] * score_resilience
    evaluation['average_score'] = {
        'display_name': 'Score',
        'weight': None,
        'value': (
            weights['comfort'] * score_comfort +
            weights['emissions'] * score_emissions +
            weights['grid_control'] * score_grid_control +
            (weighted_resilience if not np.isnan(weighted_resilience) else 0.0)
        )
    }

    return evaluation