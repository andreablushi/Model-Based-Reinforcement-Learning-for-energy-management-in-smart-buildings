import os
import sys
import hydra
from hydra import initialize, compose
import omegaconf
import numpy as np
import pandas as pd
import torch
import wandb

# Allow importing from mbrl root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mbrl.util.plot_utils import compare_kpis, plot_energy, plot_temperature
from mbrl.util.kpi_utils import evaluate_citylearn_challenge
import mbrl.algorithms.macura as macura
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.m2ac as m2ac
import mbrl.util.env as env_util
import mbrl.algorithms.sac as sac
import mbrl.util.common
from hydra.core.global_hydra import GlobalHydra
import pickle
import warnings

warnings.filterwarnings("ignore")

global agent

CHALLENGE_WEIGHTS_PHASE_1 = {
    'comfort': 0.3,
    'emissions': 0.1,
    'grid_control': 0.6,
    'resilience': 0.0
}
CHALLENGE_WEIGHTS_PHASE_2 = {
    'comfort': 0.3,
    'emissions': 0.1,
    'grid_control': 0.3,
    'resilience': 0.3
}
CHALLENGE_WEIGHTS_PHASE_CUSTOM = {
    'comfort': 0.3,
    'emissions': 0.4,
    'grid_control': 0.3,
    'resilience': 0.0
}

def run_experiment(train_cfg_name):
    """Loads a training config and runs the correct algorithm."""

    # TODO: compute TRAIN KPIs and return them
    GlobalHydra.instance().clear()
    with initialize(config_path="conf"):
        cfg = compose(config_name=train_cfg_name)

        print(f"Using algorithm: {cfg.algorithm.name}")
        env, term_fn, reward_fn = env_util.EnvHandler.make_env(cfg, test_env=False)

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        wandb.init(
            project="ModelBased",
            config=omegaconf.OmegaConf.to_container(cfg),
        )

        # Select and run algorithm
        if cfg.algorithm.name == "mbpo":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return mbpo.train(env, test_env, term_fn, cfg)

        elif cfg.algorithm.name == "m2ac":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return m2ac.train(env, test_env, term_fn, cfg)

        elif cfg.algorithm.name == "macura":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            test_env2, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return macura.train(env, test_env, test_env2, term_fn, cfg)
        
        elif cfg.algorithm.name == "sac":
            test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
            return sac.train(env, test_env, term_fn, cfg)

        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm.name}")


def test_experiment(test_cfg_name):
    """Loads a test config and evaluates the global agent."""
    global agent
    print("----------------------------------------")
    print("Testing the best agent found during training...")
    print("----------------------------------------")

    GlobalHydra.instance().clear()
    with initialize(config_path="conf"):
        cfg = compose(config_name=test_cfg_name)

        test_env, *_ = env_util.EnvHandler.make_env(cfg, test_env=True)
        # agent.act(obs)
        rl_result, rl_ep_reward = mbrl.util.common.final_evaluate(
            test_env,
            'rl',
            agent,
            cfg.seed
        )
        scores_rl = evaluate_citylearn_challenge(
            test_env,
            weights=CHALLENGE_WEIGHTS_PHASE_CUSTOM
        )

        rbc_result, rbc_ep_reward = mbrl.util.common.final_evaluate(
            test_env,
            'comfort_rbc',
            None,
            cfg.seed
        )
        scores_rbc = evaluate_citylearn_challenge(
            test_env,
            weights=CHALLENGE_WEIGHTS_PHASE_CUSTOM
        )

        print("*" * 10)
        print("RL Agent")
        for k, v in scores_rl.items():
            print(f"- {v['display_name']}: {v['value']:.4f} (weight: {v['weight']})")
        print(f"  -> Total episode reward: {rl_ep_reward:.4f}")
        print("*" * 10)

        print("Comfort RBC")
        for k, v in scores_rbc.items():
            print(f"- {v['display_name']}: {v['value']:.4f} (weight: {v['weight']})")
        print(f"  -> Total episode reward: {rbc_ep_reward:.4f}")
        print("*" * 10)

        plot_energy(rl_result, cfg.algorithm.name)
        plot_temperature(rl_result, cfg.algorithm.name)
        compare_kpis(rbc_result, rl_result, algo_names=['Comfort RBC', cfg.algorithm.name])

        workdir = os.getcwd()
        with open(os.path.join(workdir, f"{cfg.algorithm.name}_rl_results.pkl"), 'wb') as f:
            pickle.dump(rl_result, f)
        
        scores_rl_df = pd.DataFrame.from_dict(scores_rl, orient='index')
        scores_rl_df.to_csv(os.path.join(workdir, f"{cfg.algorithm.name}_rl_scores.csv"))
        scores_rbc_df = pd.DataFrame.from_dict(scores_rbc, orient='index')
        scores_rbc_df.to_csv(os.path.join(workdir, f"comfort_rbc_scores.csv"))

@hydra.main(config_path="conf", config_name="launcher_macura")
def main(launcher_cfg: omegaconf.DictConfig):
    """Top-level Hydra entrypoint (the one you call via `python -m`)."""
    global agent

    train_cfg_name = launcher_cfg.train_cfg
    test_cfg_name = launcher_cfg.test_cfg

    print(f"Launcher config:\n  train_cfg={train_cfg_name}\n  test_cfg={test_cfg_name}")

    # Run training
    best_score, best_agent = run_experiment(train_cfg_name)

    # Store globally (as in your original code)
    agent = best_agent

    # Run test
    test_experiment(test_cfg_name)


if __name__ == "__main__":
    main()