import os
import shutil
from typing import Any, List, Optional, Sequence, cast

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.models.model_env_per_build
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.replay_buffer
import mbrl.util.common
import mbrl.util.mujoco
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
from omegaconf import OmegaConf
import mbrl.util.distance_measures as dm
import colorednoise as cn
import csv

MBPO_LOG_FORMAT = [
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

def rollout_model_and_populate_sac_buffer(
        rng,
        model_env: mbrl.models.ModelEnv,
        replay_buffers: List[mbrl.util.ReplayBuffer],
        agents: List[SACAgent],
        sac_buffers: List[mbrl.util.ReplayBufferDynamicLifeTime],
        sac_samples_action: bool,
        max_rollout_length: int,
        batch_size: int,
        current_border_count: List[int],
        current_border_estimate: List[float],
        env_steps,
        num_buildings: int,
        pink_noise_exploration_mod: bool=False,
        xi:float = 1.0,
        zeta: int = 95,
):
    """Generates rollouts to create simulated trainings data for sac agent. These rollouts are used to populate the
    SAC-buffer, from which the agent can learn cheaply how to behave optimal in the approximated environment

    Args:
        model_env (mbrl.models.ModelEnv): The learned model which was transformed to behave like an environment
        replay_buffer (mbrl.util.ReplayBuffer): Replay buffer with transitions experienced in the real environment
        Used in order to sample uniformly start states for model rollouts
        agent (SACAgent): Agent which has learned a policy is used to act to get the taken actions in the rollouts
        sac_samples_action (bool): True if the agents action should be sampled according to gaussian policy
        False if the agent should just choose the mean of the gaussian
        sac_buffer (mbrl.util.ReplayBuffer) : Here the transitions of the rollouts are stored
        max_rollout_length (int): How long can the rollouts be in maximum. The real length is decided
        by the masking algorithm
        So how many actions are taken by the agent in the approximated environment
        batch_size (int): Size of batch of initial states to start rollouts and
        thus there will be batch_size*rollout_horizon more transitions stored in the sac_buffer
    """
    batches = [] * num_buildings
    for b in range(num_buildings):
        batch = replay_buffers[b].sample(batch_size)
        batches.append(batch)
    # intial_obs ndarray batchsize x observation_size
    initial_obs_parts = []
    for batch in batches:
        initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
        initial_obs_parts.append(initial_obs)
    # model_state tensor batchsize x observation_size
    model_states = model_env.reset(
        initial_obs_batch=initial_obs_parts,
        return_as_np=True,
    )
    obs = initial_obs_parts
    uncertainty_scores_for_each_rollout_sorted = [[]] * num_buildings
    number_of_certain_transitions_each_rollout = [[]] * num_buildings

    new_sac_size = [0] * num_buildings
    rollout_tracker = [np.zeros((0,))] * num_buildings
    certain_bool_map_over_all_rollouts = [np.zeros(obs[0].shape[0], dtype=bool)] *num_buildings
    border_for_this_rollout = [0.0] * num_buildings
    reduce_time = [False] * num_buildings
    
    for i in range(max_rollout_length):
        # action is of type ndarray batchsize x actionsize, action is sampled of SAC Gaussian
        actions = []
        for b in range(num_buildings):
            action = agents[b].act(obs[b], sample=sac_samples_action, batched=True)
            actions.append(action)

        # -------------------------------------------------------------------#

        # -------------------------------------------------------------------#
        # Calculate the transitions using the model
        # Make step in environment(ModelEnv->1DTransitionrewardModel->GaussianMLP->Ensemble) and get the
        # predicted obs and rewards. Also get dones and model_state.
        # For rm2ac the means and vars of the next_obs+reward are needed so get them too.
        # chosen_means,chosen_stds are of size (batchsize) and are the mean and stds of the gaussian
        # that was chosen to sample pred_next_obs, pred_rewards
        # means_of_all_ensembles, stds_of_all_ensembles are (ensemble_size x batchsize) and are all means
        # and stds of all gaussians in the ensemble
        # model_indices is (batchsize) and is the chosen model_indices of the ensemebles [0,ensemble_size)
        results = model_env.step_plus_gaussians(actions, model_states, sample=True)

        # — 3) unzippo in 9 liste —
        (next_obs_list,
        reward_list,
        term_list,
        next_model_state_list,
        means_chosen_list,
        stds_chosen_list,
        means_all_list,
        stds_all_list,
        indices_list
        ) = zip(*results)

        ensemble_size = model_env.dynamics_models[0].model.ensemble_size

        for b in range(num_buildings):
            vars_of_all_ensembles = torch.pow(stds_all_list[b], 2)
            # -------------------------------------------------------------------#


            # -------------------------------------------------------------------#

            jsp = dm.calc_pairwise_symmetric_uncertainty_for_measure_function(means_all_list[b],
                                                                                vars_of_all_ensembles,
                                                                                ensemble_size,
                                                                                dm.calc_uncertainty_score_genShen)
            uncertainty_score = jsp

            # -------------------------------------------------------------------#
            # Calculate the uncertainty threshhold. If some non zero uncertainty threshold was chosen it is used to filter
            # the generated transitions. For a zero threshold the current_border_estimate is used to filter the data,
            # it is the average over a fixed number of past border_for_this_rollout values and is given to the rollout
            # function

            if i == 0:
                zeta_percentile = np.percentile(uncertainty_score, zeta)
                border_for_this_rollout[b] = zeta_percentile * xi
                threshold = 1 / (current_border_count[b] + 1) * border_for_this_rollout[b] + current_border_count[b] / (
                            current_border_count[b] + 1) * current_border_estimate[b]
                print(f"Max Uncertainty of {zeta} percentile times {xi} factor: {border_for_this_rollout[b]}")
                print(f"Updated Uncertainty threshhold is {threshold}")
                reduce_time[b] = True
            else:
                reduce_time[b] = False

            indices_of_certain_transitions = uncertainty_score < threshold

            # certain_bool_map contains true for storing transition if it is certain enough and false else
            if i ==0:
                certain_bool_map_over_all_rollouts[b][indices_of_certain_transitions] = True
            else:
                certain_bool_map_this_rollout = np.zeros(obs[b].shape[0], dtype=bool)
                certain_bool_map_this_rollout[indices_of_certain_transitions] = True
                certain_bool_map_over_all_rollouts[b] = np.logical_and(certain_bool_map_this_rollout, certain_bool_map_over_all_rollouts[b])

            number_of_certain_transitions = certain_bool_map_over_all_rollouts[b].sum()

            rollout_tracker[b] = np.append(rollout_tracker[b], np.full((obs[b].shape[0] - number_of_certain_transitions), i))
            new_sac_size[b] = new_sac_size[b] + number_of_certain_transitions
            if number_of_certain_transitions == 0:
                endOfRollout = i
                break

            ind_sort_un = np.argsort(uncertainty_score)
            uncertainty_scores_for_each_rollout_sorted[b].append(uncertainty_score[ind_sort_un])
            number_of_certain_transitions_each_rollout[b].append(number_of_certain_transitions)
            assert np.sum(np.isinf(reward_list[b][:, 0])) == 0
            assert np.sum(np.isnan(reward_list[b][:, 0])) == 0
            # Add the filtered rollouts to the SAC Replay Buffer
            sac_buffers[b].add_batch(
                obs[b][certain_bool_map_over_all_rollouts[b]],
                actions[b][certain_bool_map_over_all_rollouts[b]],
                next_obs_list[b][certain_bool_map_over_all_rollouts[b]],
                reward_list[b][certain_bool_map_over_all_rollouts[b], 0],# pred_rewards and v
                term_list[b][certain_bool_map_over_all_rollouts[b], 0],# need to be of size batchsize not batchsize x 1
                term_list[b][certain_bool_map_over_all_rollouts[b],0], #Let it be false all the time because model predictions do no get truncated
                reduce_time=reduce_time[b] #is true for i==0 and serves the purpose to reduce the lifetime of the stored items in replay buffer
            )
            # squeezing to transform pred_terminateds from batch_size x 1 to batchsize
            certain_bool_map_over_all_rollouts = np.logical_and(~(term_list[b].squeeze()),
                                                                certain_bool_map_over_all_rollouts)
        obs = next_obs_list
        model_state = model_env.reset(
            initial_obs_batch=cast(np.ndarray, obs),
            return_as_np=True,
        )

    return new_sac_size, border_for_this_rollout

def compute_batch_mean_rewards(
    work_dir: str,
    env_step: int,
    num_buildings: int,
) -> None:
    """
    Read the csv with the single rewards from the single agents and compute 
    the mean batch reward across all buildings for comparison with other algorithms
    """
    building_dirs = [
        os.path.join(work_dir, f"building_{i}")
        for i in range(num_buildings)
    ]
    overall_csv_path = os.path.join(work_dir, "overall_batch_rewards.csv")

    # Collect all rewards for this step
    rewards = []
    
    for b, bdir in enumerate(building_dirs):
        csv_path = os.path.join(bdir, "train.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping building {b}")
            continue
            
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row_step = int(row["step"])
                except (KeyError, ValueError):
                    continue
                    
                if row_step == env_step:
                    try:
                        reward = float(row["batch_reward"])
                        rewards.append(reward)
                    except (KeyError, ValueError):
                        print(f"Warning: Invalid batch_reward value in {csv_path} at step {env_step}")
                    break

    # Calculate statistics
    count = len(rewards)
    if count > 0:
        mean_reward = sum(rewards) 
        min_reward = min(rewards)
        max_reward = max(rewards)
    else:
        mean_reward = min_reward = max_reward = 0.0

    # Write to output CSV
    write_header = not os.path.exists(overall_csv_path)
    with open(overall_csv_path, "a", newline="") as fout:
        writer = csv.writer(fout)
        if write_header:
            writer.writerow(["step", "batch_mean_reward", "min_reward", "max_reward", "num_buildings"])
        writer.writerow([env_step, mean_reward, min_reward, max_reward, count])

    print(f"Step {env_step}: mean={mean_reward:.6f}, min={min_reward:.6f}, max={max_reward:.6f}, buildings={count}")
def change_capacity_replay_buffer(
        sac_buffer: Optional[mbrl.util.ReplayBufferDynamicLifeTime],
        obs_shape: Sequence[int],
        act_shape: Sequence[int],
        new_capacity: int,
        seed: int,
        lifetime: int,
) -> mbrl.util.ReplayBufferDynamicLifeTime:
    """If the given sac_buffer is None, a new ReplayBuffer is created.
    Else the existing sac_buffers size is changed, existing data will be kept.

    Args:
        sac_buffer (mbrl.util.ReplayBuffer): Given replay_buffer which size should be changed to new_capacity.
        If None then new buffer will be created with obs_shape and act_shape as transition dimensions
        and new_capacity as number of transitions which can be stored
        obs_shape (Sequence[int]): Shape of observation and next_observation in transition
        act_shape (Sequence[int]): Shape of action in transition
        new_capacity (int): How many transitions can now be stored in buffer
        lifetime (int): number of rollouts until data is deleted

    Returns:
        (float): The average reward of the num_episode episodes
    """
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        # sac buffer needs to be created
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
            new_buffer = mbrl.util.ReplayBufferDynamicLifeTime(new_capacity, obs_shape, act_shape, lifetime, rng=rng)
            return new_buffer
        # capacity needs to be increased
        else:
            rng = sac_buffer.rng
            new_buffer = mbrl.util.ReplayBufferDynamicLifeTime(new_capacity, obs_shape, act_shape, lifetime, rng=rng)
            obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
            new_buffer.add_batch(obs, action, next_obs, reward, done)
            return new_buffer
    return sac_buffer

def train(
        env: gym.Env,
        test_env: gym.Env,
        distance_env: gym.Env,
        termination_fn: mbrl.types.TermFnType,
        cfg: omegaconf.DictConfig,
        silent: bool = False,
        work_dir: Optional[str] = None,
) -> np.float32:
    """ This is the starting point for the mbpo algorithm. We will learn on the env environment and test agents
    performance on test_env. We interchange model_training and agent_training. The model is trained using experienced
    trajectories in the real environment using the current agent. After that the agent is trained using artificial
    roulouts using the learned model.

    Args:
        env (gym.Env): The environment used to learn the model
        test_env (gym.Env): The environment used to evaluate the model and the agent after each epoch
        It seems to be the same es env.
        distance_env (gym.Env): The environment used to track the real next states for rollouts
        termination_fn (mbrl.types.TermFnType): Function which returns if state is terminal state or not
        cfg (omegaconf.DictConfig): Complete configuration of algorithm
        See mbpo_cfg_explained.txt for configuration details.
        silent (bool): True if the log should output something or false if not
        work_dir (Optional[str]) The current working directory

    Returns:
        (float): Best reward after evaluation
    """


    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # ------------------- Initialization -------------------

    if work_dir == None:
        print("Running FMACURA algorithm from a fresh start!")
        work_dir = os.getcwd()

    # Cleaning the work directory
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    # ------------------- Create Logger -------------------
    common_logger = mbrl.util.Logger(work_dir)
    common_logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        mbrl.constants.MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    # ------------------- Create Viderecorder -------------------
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)



    # --- Separation of the environment ---
    logger_updates_made = 0

    # It is assumed that the environment is a CityLearn environment and that it has a certain number of buildings
    # It needs to be generalized to other environments, or number of building
    num_buildings = len(env.unwrapped.buildings)

    # Create a work directory for each building
    for b in range(num_buildings):
        building_dir = os.path.join(work_dir, f"building_{b}")
        os.makedirs(building_dir, exist_ok=True)


    full_obs_shape = env.observation_space.shape      
    full_act_shape = env.action_space.shape  

    obs_dim_per_building = full_obs_shape[-1] if len(full_obs_shape) > 1 else full_obs_shape[0] // num_buildings
    act_dim_per_building = full_act_shape[-1] if len(full_act_shape) > 1 else full_act_shape[0] // num_buildings

    max_rollout_length = cfg.algorithm.max_rollout_length

    # Creation of a list of SAC Agents, models, and replay buffers for each building
    agents: List[SACAgent] = []
    real_replay_buffers: List[mbrl.util.ReplayBuffer] = []
    dynamics_models: List[Any] = []
    model_trainers: List[mbrl.models.ModelTrainer] = []
    loggers: List[Any] = []

    # Complete common configuration for the agent
    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)

    # Unify randomness
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    obs_shape = (obs_dim_per_building,)
    act_shape = (act_dim_per_building,)
    action_space_box = gym.spaces.Box(0.0, 1.0, shape=(act_dim_per_building,), dtype=np.float32)

    # Initialize the components for each building
    for b in range(num_buildings):
        # Indipendent logger
        loggers.append(mbrl.util.Logger(os.path.join(work_dir, f"building_{b}")))
        loggers[b].register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        mbrl.constants.MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
        )
        # 1) SAC Agent
        # Check the need for a separate cfg for each agent
        single_agent = SACAgent(
            pytorch_sac_pranz24.SAC(
                obs_dim_per_building,
                action_space_box,
                cfg.algorithm.agent.args
            )
        )
        agents.append(single_agent)

        # Common configuration for the dtype
        use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
        dtype = np.double if use_double_dtype else np.float32
        # 2) Replay Buffer Environment
        real_rb = mbrl.util.common.create_replay_buffer(
            cfg,
            (obs_dim_per_building,),            
            (act_dim_per_building,),            
            rng=rng,  
            obs_type=dtype,
            action_type=dtype,
            reward_type=dtype,
        )
        real_replay_buffers.append(real_rb)

        # 3) Model Environment
        dyn_model_b = mbrl.util.common.create_one_dim_tr_model(
            cfg,
            (obs_dim_per_building,),
            (act_dim_per_building,)
        )
        dynamics_models.append(dyn_model_b)

 

        trainer_b = mbrl.models.ModelTrainer(
            dyn_model_b,
            optim_lr=cfg.overrides.model_lr,
            weight_decay=cfg.overrides.model_wd,
            logger=loggers[b]
        )
        model_trainers.append(trainer_b)

    # It take all the enviroment, but it will use only one building
    model_env = mbrl.models.model_env_per_build.ModelEnvPerBuild(
        env,                     
        dynamics_models,
        termination_fn,
        None,
        generator=torch.Generator(device=cfg.device)
    )
    real_experienced_states_full = []
    # -------------- Gather initial data using random or random initialized SAC-Agent --------------
    mbrl.util.common.rollout_agent_trajectories_per_building(
        real_experienced_states_full,
        env,
        cfg.algorithm.initial_exploration_steps,
        agents,
        {},
        trial_length=None,
        callback=None,
        replay_buffers=real_replay_buffers,
    )

    # ---------------------------------------------------------
    # --------------------- Start Training---------------------
    # ---------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # --------------------- Initialization before training starts ---------------------
    # ---------------------------------------------------------------------------------
    # number of effective added transitions per environment step, but sac_buffer is filled only after each freq_train_model env_steps
    effective_model_rollouts_per_step = cfg.overrides.effective_model_rollouts_per_step
    freq_train_model = cfg.algorithm.freq_train_model
    epoch_length = cfg.overrides.epoch_length
    rollout_batch_size = effective_model_rollouts_per_step * freq_train_model
    num_epochs_to_retain_sac_buffer = cfg.overrides.num_epochs_to_retain_sac_buffer
    num_sac_updates_per_step = cfg.overrides.num_sac_updates_per_step
    sac_updates_every_steps = cfg.overrides.sac_updates_every_steps
    real_data_ratio = cfg.algorithm.real_data_ratio
    sac_batch_size = cfg.overrides.sac_batch_size

    #Network reset
    critic_reset = cfg.algorithm.critic_reset
    critic_reset_every_step = cfg.algorithm.critic_reset_every_step
    critic_reset_factor = cfg.algorithm.critic_reset_factor


    unc_tresh_run_avg_history = cfg.overrides.unc_tresh_run_avg_history
    pink_noise_exploration_mod = cfg.overrides.pink_noise_exploration_mod
    xi = cfg.overrides.xi
    zeta = cfg.overrides.zeta


    sac_buffers = [None] * num_buildings
    best_eval_reward = -np.inf
    updates_made = 0
    # real steps taken in environment
    env_steps = 0
    # full model and agent training phase
    epoch = 0
    total_max_steps_in_environment = cfg.overrides.num_steps

    # we will stop training after we reach our final steps in environment counting over all epochs
    current_border_count_position = [0] * num_buildings
    current_border_count = [0] * num_buildings
    current_border_estimate = [0] * num_buildings
    current_border_estimate_list_full = [False] * num_buildings
    # Number of maximum border_for_this_rollout_values to average to get the uncertainty threshold
    Max_Count = unc_tresh_run_avg_history
    # Max_Count = int(total_max_steps_in_environment / 5000 * (epoch_length / freq_train_model))
    # here are these values safed
    current_border_estimate_list = [np.empty(Max_Count)] * num_buildings
    while env_steps < total_max_steps_in_environment:
        # ---------------------------------------------------------------------------------
        # --------------------- Initialization for new epoch ---------------------
        # ---------------------------------------------------------------------------------
        # Because cfg.overrides.freq_train_model cancels out two ways to calculate the capacity
        # sac_buffer_capacity = max_rollout_length * effective_model_rollouts_per_step * epoch_length
        # sac_buffer_capacity = max_rollout_length * rollout_batch_size * trains_per_epoch

        # Common proprietes for all agent
        trains_per_epoch = epoch_length / freq_train_model
        sac_buffer_capacity = max_rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity = sac_buffer_capacity * num_epochs_to_retain_sac_buffer
        sac_buffer_capacity = int(sac_buffer_capacity)
        lifetime = num_epochs_to_retain_sac_buffer * trains_per_epoch
        sac_buffer_capacity = int(sac_buffer_capacity)
        # Initialize the sac buffer for all the agent
        for b in range(num_buildings): 
            sac_buffers[b] = change_capacity_replay_buffer(
                sac_buffers[b], obs_shape, act_shape, sac_buffer_capacity, cfg.seed, lifetime
            )

        obs, terminated, truncated = None, False, False
        for steps_epoch in range(epoch_length):
            if steps_epoch == 0 or (terminated or truncated):
                obs, info = env.reset()
                terminated = truncated =  False
            # --- Doing env step and adding to model dataset ---

            next_obs, reward, terminated, truncated, _ = mbrl.util.common.step_env_and_add_to_buffer_per_building(
                    env, obs, agents, {}, replay_buffers=real_replay_buffers,num_buildings=num_buildings,
            )

            env_steps += 1

            # --------------- Model Training -----------------
            # in each epoch all cfg.overrides.freq_train_model the model is trained and the sac_buffer is filed
            if env_steps % freq_train_model == 0:

                for b in range(num_buildings):
                    # Start Model Training
                    print("BUILDING NUMBER ", b)
                    mbrl.util.common.train_model_and_save_model_and_data(
                        dynamics_models[b], model_trainers[b], cfg.overrides, real_replay_buffers[b], work_dir=os.path.join(work_dir, f"building_{b}")
                    )
                # --------- Rollout new model and store imagined trajectories --------
                # generates maximally rollout_length * rollout_batch_size
                # (=freq_train_model * effective_model_rollouts_per_step) new transitions for SAC buffer
                new_sac_size, current_border_estimate_update = rollout_model_and_populate_sac_buffer(
                    rng,
                    model_env,
                    real_replay_buffers,
                    agents,
                    sac_buffers,
                    cfg.algorithm.sac_samples_action,
                    max_rollout_length,
                    rollout_batch_size,
                    current_border_count,
                    current_border_estimate,
                    env_steps,
                    num_buildings,
                    pink_noise_exploration_mod,
                    xi,
                    zeta
                )
                current_border_estimate_list[b][current_border_count_position[b]] = current_border_estimate_update[b]
                if current_border_estimate_list_full[b] == False:
                    if current_border_count_position[b] == Max_Count - 1:
                        current_border_estimate_list_full[b] = True
                        current_border_count[b] = Max_Count - 1
                    else:
                        current_border_count[b] = current_border_count_position[b] + 1
                current_border_count_position[b] = (current_border_count_position[b] + 1) % Max_Count
                if current_border_estimate_list_full[b]:
                    current_border_estimate[b] = np.mean(current_border_estimate_list[b])
                else:
                    current_border_estimate[b] = np.mean(current_border_estimate_list[b][0:current_border_count_position[b]])
                # --------------- Agent Training -----------------
                # here is a formula which controlls learning steps proportionally to the filling of the SAC buffer

            # My approximation for dynamic updates per step
            total_num_stored = sum(buf.num_stored for buf in sac_buffers)
            print("Total num stored: ",total_num_stored)
            total_capacity   = sum(buf.capacity   for buf in sac_buffers) 
            print(total_capacity)
            print(sac_buffers[0].capacity)
            overall_ratio = (total_num_stored) / total_capacity if total_capacity > 0 else 0.0  
            dynamic_updates_per_step = int(overall_ratio * num_sac_updates_per_step)    
            #Periodic network reset
            
            if critic_reset and env_steps%critic_reset_every_step==0:
                for b in range(num_buildings):
                    agents[b].sac_agent.critic.reset_weights(critic_reset_factor)

            print("Dynamic updates: ", dynamic_updates_per_step)
            for _ in range(10):
                print("Updating")
                for b in range(num_buildings):
                    use_real_data = rng.random() < real_data_ratio
                    which_buffer = real_replay_buffers[b] if use_real_data else sac_buffers[b]
                    if env_steps % sac_updates_every_steps != 0 or len(which_buffer) < sac_batch_size:
                        print("BREAK")
                        break  # only when buffer is full enough to batch start training
                    print("Correct size")
                    agents[b].sac_agent.update_parameters(
                        which_buffer,
                        cfg.overrides.sac_batch_size,
                        updates_made,
                        loggers[b],
                        reverse_mask=True,
                    )
                updates_made+=1
                if not silent and updates_made % cfg.log_frequency_agent == 0 and updates_made!=0 :
                    for b in range(num_buildings):
                        loggers[b].dump(updates_made, save=True)
                    compute_batch_mean_rewards(work_dir,updates_made,num_buildings)




            # ------ Epoch ended (evaluate and save model) ------

            if env_steps % epoch_length == 0:
                print(f"Epoch ended - env-steps:{env_steps}")
                avg_reward = mbrl.util.common.evaluate_for_building(
                    test_env, agents, cfg.algorithm.num_eval_episodes, video_recorder
                )
                common_logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps - 1,
                        "episode_reward": avg_reward,
                        "rollout_length": max_rollout_length,
                    },
                )
                if save_video:
                    video_recorder.save(f"{epoch}.mp4")
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    for b in range(num_buildings):
                        # Save the agents
                        agents[b].sac_agent.save_checkpoint(
                            ckpt_path=os.path.join(work_dir, f"sac_{b}.pth")
                        )
                epoch += 1
            obs = next_obs
    return np.float32(best_eval_reward)