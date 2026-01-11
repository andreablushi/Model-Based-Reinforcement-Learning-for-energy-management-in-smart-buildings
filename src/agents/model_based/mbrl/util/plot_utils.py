import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_energy(
    res,
    suffix=''
):
    """
    Generate and save a comprehensive energy management visualization with four subplots.
    This function creates a detailed energy profile analysis plot showing building consumption
    components, demand vs. generation, battery control signals, and state of charge over time.
    Parameters
    ----------
    res : dict
        A nested dictionary containing environment and device data with the following structure:
        - res['env_h']['cooling_device']['consumption'] : array-like
            Cooling device power consumption [kW]
        - res['env_h']['dhw']['consumption'] : array-like
            Domestic hot water device power consumption [kW]
        - res['env_h']['non_shiftable_load'] : array-like
            Non-shiftable building load [kW]
        - res['env_h']['battery']['consumption'] : array-like
            Battery charging power [kW]
        - res['env_h']['solar_generation'] : array-like
            Photovoltaic generation [kW]
        - res['env_h']['battery']['discharge'] : array-like
            Battery discharge/charge control signal [kW/h]
        - res['env_h']['battery']['soc'] : array-like
            Battery state of charge [%]
        - res['env_h']['net_electricity_consumption'] : array-like
            Net electricity consumption from grid [kW]
    suffix : str, optional
        Suffix appended to the output filename (default is empty string '')
    Returns
    -------
    None
        The function saves the generated plot to disk as 'energy_profile_{suffix}.png'
    Notes
    -----
    The function generates four stacked/line plots:
    1. Building consumption components (stacked area) with total demand overlay
    2. Building demand, PV generation, and net load comparison
    3. Battery (dis)charge control signal
    4. Battery state of charge over time
    The plot includes an explanatory annotation clarifying net load interpretation
    (positive = grid import, negative = grid export).
    """
    print("*"*50+'\n')
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
    path = os.path.join(os.getcwd(), f'energy_profile_{suffix}.png')
    plt.savefig(
        path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def compare_kpis(res_1, res_2, algo_names=[]):
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
    path = os.path.join(os.getcwd(), 'kpi_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_temperature(results, suffix=''):
    """
    Plot temperature profiles including indoor temperature, outdoor temperature, 
    and comfort band visualization.
    This function creates a comprehensive temperature analysis plot showing:
    - Indoor temperature over time
    - Outdoor temperature over time
    - Comfort band (setpoint ± tolerance range)
    Parameters
    ----------
    results : dict
        Dictionary containing simulation results with nested structure:
        results['env_h']['temperature'] should contain:
        - 'indoor_temperature' : array-like
            Indoor temperature values
        - 'indoor_temperature_set_point' : array-like
            Target setpoint temperature values
        - 'outdoor_temperature' : array-like
            Outdoor temperature values
        - 'comfort_band' : float
            Temperature tolerance band around setpoint (±value)
    suffix : str, optional
        Suffix to append to the output filename (default: '')
    Returns
    -------
    None
        Saves the plot as 'temperature_profile_{suffix}.png'
    Notes
    -----
    - Uses seaborn styling with "whitegrid" style and "talk" context
    - Uses colorblind-friendly palette
    - Saves output at 300 dpi with tight bounding box
    - Figure size: 30x10 inches
    """

    temp_1 = results['env_h']['temperature']
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
    path = os.path.join(os.getcwd(), f'temperature_profile_{suffix}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')