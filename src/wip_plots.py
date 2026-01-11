

from agents.model_based.mbrl.util.CityLearnWrappers import CityLearnKPIWrapper
from agents.model_based.mbrl.util.env import CityLearnSchema
from rewards.CityLearnReward import SolarPenaltyAndComfortReward
from citylearn.citylearn import CityLearnEnv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_energy(
    time,
    cooling_device_consumption,
    dhw_device_consumption,
    non_shiftable_load,
    battery_charge,
    pv_generation,
    battery_action,
    battery_soc,
    net_load,
    battery_power_capacity=None,
    index=0
):
    """
    Visualizes the building energy flows and battery behavior over time.

    This function produces a four-panel plot showing:
    1. Building electricity consumption components (stacked area plot)
    2. Building demand, PV generation, and resulting net load
    3. Battery control signal or real power (depending on scaling)
    4. Battery state of charge (SoC)

    The color scheme is colorblind-friendly and legends are positioned on the right.
    A note is also included to explain the meaning of positive/negative net load.

    Parameters
    ----------
    time : array-like
        Time steps or timestamps corresponding to simulation periods.
    cooling_device_consumption : array-like
        Electricity consumption of the cooling device [kW].
    dhw_device_consumption : array-like
        Electricity consumption for domestic hot water [kW].
    non_shiftable_load : array-like
        Electricity demand from non-controllable devices [kW].
    battery_charge : array-like
        Battery charging power [kW]; positive for charging.
    pv_generation : array-like
        PV electricity generation [kW]; positive values represent generation.
    battery_action : array-like
        Normalized battery control signal in [-1, 1].
    battery_soc : array-like
        Battery state of charge (SoC) [%].
    net_load : array-like
        Net building load [kW], where positive means importing from the grid,
        and negative means exporting to the grid.
    battery_power_capacity : float, optional
        Maximum charge/discharge power of the battery [kW].
        If provided, the action will be scaled by this value.
    index : int, optional
        Building index or identifier for labeling purposes.

    Returns
    -------
    None
        The function saves a PNG file of the figure in the current directory.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 6)

    # --- Prepare data ---
    pv_generation = -1 * pv_generation  # Flip sign for plotting
    pv_generation = pv_generation[:len(time)]
    battery_soc = battery_soc[:len(time)]
    dhw_device_consumption = dhw_device_consumption[:len(time)]

    # --- Derived quantities ---
    building_demand = (
        cooling_device_consumption
        + dhw_device_consumption
        + non_shiftable_load
        + battery_charge
    )

    if battery_power_capacity is not None:
        battery_power = battery_action * battery_power_capacity
        label_action = "Battery Power [kW]"
    else:
        battery_power = battery_action
        label_action = "Battery Action [-1, 1]"

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
        f'building_{index}_energy_profile.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()



def plot_temperature(
    time,
    indoor_temps,
    outdoor_temps,
    setpoints,
    cooling_action,
    building_id,
    cooling_power_capacity=None
):
    """
    Plots indoor/outdoor temperatures, setpoints, and cooling device control signal in two subfigures.

    The first subplot shows temperature dynamics (indoor, outdoor, setpoint),
    while the second shows the cooling control signal or power, depending on scaling.

    Parameters
    ----------
    time : array-like
        Time steps.
    indoor_temps : array-like
        Indoor air temperature [°C].
    outdoor_temps : array-like
        Outdoor temperature [°C].
    setpoints : array-like
        Setpoint temperature [°C].
    cooling_action : array-like
        Normalized cooling control signal [-1, 1] or cooling power [kW].
    building_id : int
        Building index or identifier for labeling purposes.
    cooling_power_capacity : float, optional
        Maximum cooling power [kW]. If provided, scales the cooling action.

    Returns
    -------
    None
        The function saves a PNG file of the figure in the current directory.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")
    palette = sns.color_palette("colorblind", 5)

    # Ensure arrays match in length
    n = len(indoor_temps)
    time = time[:n]
    outdoor_temps = outdoor_temps[:n]
    setpoints = setpoints[:n]
    cooling_action = cooling_action[:n]

    # Optional scaling
    if cooling_power_capacity is not None:
        cooling_power = cooling_action * cooling_power_capacity
        label_action = "Cooling Power [kW]"
    else:
        cooling_power = cooling_action
        label_action = "Cooling Action [-1, 1]"

    # --- Create figure and subplots ---
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3, right=0.85)

    # 0️⃣ Temperature profiles
    axs[0].plot(time, indoor_temps, label='Indoor Temperature', color=palette[0], lw=2)
    axs[0].plot(time, outdoor_temps, label='Outdoor Temperature', color=palette[1], lw=2)
    axs[0].plot(time, setpoints, label='Setpoint Temperature', color=palette[2], linestyle='--', lw=2)
    axs[0].set_ylabel('Temperature [°C]')
    axs[0].set_title(f'Building {building_id} - Temperature Profiles')
    axs[0].legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # 1️⃣ Cooling Control Signal
    axs[1].axhline(0, color='black', lw=0.8, linestyle='--')
    sns.lineplot(x=time, y=cooling_power, ax=axs[1], color=palette[3], lw=1.8)
    axs[1].set_ylabel(label_action)
    axs[1].set_xlabel('Time')
    axs[1].set_title('Cooling Device Control Signal')

    plt.tight_layout()
    plt.savefig(f'building_{building_id}_temperature_profile.png', dpi=300, bbox_inches='tight')
    # plt.show()


env_name = "citylearn_challenge_2023_phase_1"
eval_env = CityLearnEnv(env_name, central_agent=True)
rf = SolarPenaltyAndComfortReward(eval_env.schema)
eval_env.reward_function = rf
eval_env.random_seed=0
eval_env = CityLearnKPIWrapper(eval_env)

# Number of the chosen building and number of episode trainig
num_episodes = 1

def make_plots(eval_env, cooling_actions, battery_actions, dhw_actions):
     for i in range(len(eval_env.unwrapped.buildings)):
        indoor_temps = eval_env.unwrapped.buildings[i].indoor_dry_bulb_temperature
        outdoor_temps = eval_env.unwrapped.buildings[i].weather.outdoor_dry_bulb_temperature
        setpoints = eval_env.unwrapped.buildings[i].indoor_dry_bulb_temperature_cooling_set_point
        solar_generation = eval_env.unwrapped.buildings[i]._Building__solar_generation
        battery_consumption = eval_env.unwrapped.buildings[i].electrical_storage_electricity_consumption # Positive when charging, negative when discharging
        cooling_device_consumption = eval_env.unwrapped.buildings[i].cooling_electricity_consumption 
        dhw_device_consumption = eval_env.unwrapped.buildings[i].dhw_device._ElectricDevice__electricity_consumption
        non_shiftable_load = eval_env.unwrapped.buildings[i].non_shiftable_load
        net_load = eval_env.unwrapped.buildings[i].net_electricity_consumption
        battery_soc = eval_env.unwrapped.buildings[i].electrical_storage.soc
        dhw_demand = eval_env.unwrapped.buildings[i].dhw_demand

        # battery_consumption
        # Only consumption (no PV generation, no discharging)
        battery_charge = np.maximum(0.0, battery_consumption)

        plot_energy(
            time=np.arange(len(indoor_temps)),
            cooling_device_consumption=cooling_device_consumption,
            dhw_device_consumption=dhw_device_consumption,
            non_shiftable_load=non_shiftable_load,
            battery_charge=battery_charge,
            pv_generation=solar_generation,
            battery_action=battery_actions[i],
            battery_soc=battery_soc,
            net_load=net_load,
            index=i
        )

        plot_temperature(
            time=np.arange(len(indoor_temps)),
            indoor_temps=indoor_temps,
            outdoor_temps=outdoor_temps,
            setpoints=setpoints,
            cooling_action=cooling_actions[i],
            building_id=i
        )

        # plot_dhw(
        #     time=np.arange(len(indoor_temps)),
        #     dhw_demand=dhw_demand,
        #     dhw_action=dhw_actions[i],
        #     building_id=i
        # )

# Evaluate
infos = []
for episode in range(num_episodes):
    observations, _ = eval_env.reset(seed=episode)
    terminated = truncated = False
    idx_battery_action = np.where(np.array(eval_env.unwrapped.buildings[0].active_actions) == "electrical_storage")[0][0]
    idx_cooling_action = np.where(np.array(eval_env.unwrapped.buildings[0].active_actions) == "cooling_device")[0][0]
    idx_dhw_action = np.where(np.array(eval_env.unwrapped.buildings[0].active_actions) == "dhw_storage")[0][0]
    battery_actions = [[0.0] for _ in range(len(eval_env.unwrapped.buildings))]
    cooling_actions = [[0.0] for _ in range(len(eval_env.unwrapped.buildings))]
    dhw_actions = [[0.0] for _ in range(len(eval_env.unwrapped.buildings))]

    for _ in range(100):
        actions = eval_env.action_space.sample()
        observations, reward, terminated, truncated, info = eval_env.step(actions)
        for i in range(len(eval_env.unwrapped.buildings)):
            battery_actions[i].append(actions[3*i + idx_battery_action])
            cooling_actions[i].append(actions[3*i + idx_cooling_action])
            dhw_actions[i].append(actions[3*i + idx_dhw_action])

    make_plots(eval_env, cooling_actions=cooling_actions, battery_actions=battery_actions, dhw_actions=dhw_actions)
    print("INFO", info)

# INDOOR TEMP
# eval_env.unwrapped.buildings[0].indoor_dry_bulb_temperature
# OUTDOOR TEMP
# eval_env.unwrapped.buildings[0].weather.outdoor_dry_bulb_temperature
# SETPOINT
# eval_env.unwrapped.buildings[0].indoor_dry_bulb_temperature_cooling_set_point