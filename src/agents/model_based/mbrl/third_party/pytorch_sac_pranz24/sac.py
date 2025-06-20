import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from mbrl.third_party.pytorch_sac_pranz24.model import (
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
    QNetworkLN,
    GaussianPolicyLN,
    DeterministicPolicyLN
)
from mbrl.third_party.pytorch_sac_pranz24.utils import hard_update, soft_update
import wandb

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.layernorm = args.layernorm
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device
        if self.layernorm:
            self.critic = QNetworkLN(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = QNetworkLN(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)


        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None:
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            if self.layernorm:
               self.policy = GaussianPolicyLN(
                    num_inputs, action_space.shape[0], args.hidden_size, action_space
                ).to(self.device)
            else:
                self.policy = GaussianPolicy(
                    num_inputs, action_space.shape[0], args.hidden_size, action_space
                ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if self.layernorm:
                self.policy = DeterministicPolicyLN(
                    num_inputs, action_space.shape[0], args.hidden_size, action_space
                ).to(self.device)
            else:
                self.policy = DeterministicPolicy(
                    num_inputs, action_space.shape[0], args.hidden_size, action_space
                ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def select_action_eps(self, state, eps, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        eps = torch.FloatTensor(eps)
        if not batched:
            state = state.unsqueeze(0)
            eps = eps.unsqueeze(0)
        state = state.to(self.device)
        eps = eps.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample_using_eps(state, eps)
        else:
            _, _, action = self.policy.sample_using_eps(state, eps)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def update_parameters(
        self, memory, batch_size, updates, logger=None, reverse_mask=False
    ):
        # Sample a batch from memory and ignore truncated transititons
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch, #these corresponds to the terminated ones
            _, #truncated not used
        ) = memory.sample(batch_size).astuple()
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        if logger is not None:
            logger.log("train/batch_reward", reward_batch.mean(), updates)
            logger.log("train/batch_reward_max", reward_batch.max(), updates)
            logger.log("train/batch_reward_min", reward_batch.min(), updates)
            logger.log("train_critic/loss", qf_loss, updates)
            logger.log("train_critic/loss_1", qf1_loss, updates)
            logger.log("train_critic/loss_2", qf2_loss, updates)
            logger.log("train_actor/loss", policy_loss, updates)
            logger.log("train_actor/qf1M", qf1.max(), updates)
            logger.log("train_actor/qf2M", qf2.max(), updates)
            logger.log("train_actor/qtargetM", min_qf_next_target.max(), updates)
            logger.log("train_actor/nextQvalue", next_q_value.max(), updates)
            logger.log("train_actor/masked", mask_batch.min(), updates)
            if self.automatic_entropy_tuning:
                logger.log("train_actor/target_entropy", self.target_entropy, updates)
            else:
                logger.log("train_actor/target_entropy", 0, updates)
            logger.log("train_actor/entropy", -log_pi.mean(), updates)
            logger.log("train_alpha/loss", alpha_loss, updates)
            logger.log("train_alpha/value", self.alpha, updates)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )
    

    def update_parameters_per_building(
        self, memory, batch_size, updates, work_dir, num_building, log_reward, logger=None, reverse_mask=False
    ):
        # Sample a batch from memory and ignore truncated transititons
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch, #these corresponds to the terminated ones
            _, #truncated not used
        ) = memory.sample(batch_size).astuple()
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # if i need to log rewards per building
        if log_reward:
            reward_list = reward_batch.squeeze(1).cpu().tolist()
            csv_path = os.path.join(work_dir, "single_rewards.csv")
            write_header = not os.path.exists(csv_path)
            import csv
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["build","step", "reward"])
                for r in reward_list:
                    writer.writerow([num_building,updates, r])

        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        if logger is not None:
            logger.log("train/batch_reward", reward_batch.mean(), updates)
            logger.log("train/batch_reward_max", reward_batch.max(), updates)
            logger.log("train/batch_reward_min", reward_batch.min(), updates)
            logger.log("train_critic/loss", qf_loss, updates)
            logger.log("train_critic/loss_1", qf1_loss, updates)
            logger.log("train_critic/loss_2", qf2_loss, updates)
            logger.log("train_actor/loss", policy_loss, updates)
            logger.log("train_actor/qf1M", qf1.max(), updates)
            logger.log("train_actor/qf2M", qf2.max(), updates)
            logger.log("train_actor/qtargetM", min_qf_next_target.max(), updates)
            logger.log("train_actor/nextQvalue", next_q_value.max(), updates)
            logger.log("train_actor/masked", mask_batch.min(), updates)
            if self.automatic_entropy_tuning:
                logger.log("train_actor/target_entropy", self.target_entropy, updates)
            else:
                logger.log("train_actor/target_entropy", 0, updates)
            logger.log("train_actor/entropy", -log_pi.mean(), updates)
            logger.log("train_alpha/loss", alpha_loss, updates)
            logger.log("train_alpha/value", self.alpha, updates)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    # Save model parameters
    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
