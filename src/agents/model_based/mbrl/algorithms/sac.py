import os
import gymnasium as gym
import numpy as np
import torch
import omegaconf
from typing import Optional

import mbrl.constants
import mbrl.types
import mbrl.util.common as common
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac_pranz24 import SAC
from mbrl.third_party.pytorch_sac import VideoRecorder
import mbrl.util.mujoco
from omegaconf import OmegaConf

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("env_step", "S", "int"),
]

def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    """
    Train a base SAC agent on the specified Gym environment (no model-based rollouts).
    """
    print("Using SAC Final")
    # Complete agent config and instantiate SAC agent
    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    sac_impl = SAC(
        cfg.algorithm.agent.num_inputs,
        env.action_space,
        cfg.algorithm.agent.args,
    )
    agent = SACAgent(sac_impl)

    # Setup work directory
    if work_dir is None:
        work_dir = os.getcwd()
        load_checkpoints = False
    else:
        load_checkpoints = True
    os.makedirs(work_dir, exist_ok=True)

    # Logger and video recorder
    logger = mbrl.util.Logger(work_dir)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    video_dir = work_dir if cfg.get("save_video", False) else None
    video_recorder = VideoRecorder(video_dir)

    # RNG seeds
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # Replay buffer
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
        

    # -------------- Gather initial data using random or random initialized SAC-Agent --------------
    updates_made = 0
    env_steps = 0
    best_eval_reward = -np.inf
    epoch = 0

    # Main training loop
    while env_steps < cfg.overrides.num_steps:
        obs, terminated, truncated = None, False, False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or terminated or truncated:
                obs, _ = env.reset()
                terminated = False
                truncated = False
            # Select action
            if cfg.algorithm.initial_exploration_steps > env_steps:
                action = agent.act(obs, sample=True, batched=False)
            else:
                action = agent.sac_agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)

            # SAC updates
            if len(replay_buffer) > cfg.overrides.sac_batch_size:
                for _ in range(cfg.overrides.num_sac_updates_per_step):
                    agent.sac_agent.update_parameters(
                            replay_buffer,
                            cfg.overrides.sac_batch_size,
                            updates_made,
                            logger,
                    )
                    updates_made += 1
                    if updates_made % cfg.log_frequency_agent == 0:
                        logger.dump(updates_made, save=True)


            # End of evaluation interval
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = common.evaluate(
                    test_env,
                    agent,
                    cfg.algorithm.num_eval_episodes,
                    video_recorder,
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {"epoch": epoch, "env_step": env_steps + 1, "episode_reward": avg_reward, "rollout_length": 0,},
                )
                logger.dump(updates_made, save=True)
                # Save best model
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(ckpt_path=os.path.join(work_dir, "sac.pth"))
                epoch += 1

            # Increment step counter
            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
