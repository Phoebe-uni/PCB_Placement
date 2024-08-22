import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

import utils
import tracker
import time

#Implement PPO
#Writen by YYF.



class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 pi : list = [400, 300],
                 activation_fn: str = "relu",
                 device: str = "cuda"
                 ):

        super(Actor, self).__init__()
        self.device = device
        self.pi = nn.Sequential()
        in_size = state_dim
        for layer_sz in pi:
            self.pi.append(nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        self.pi.append(nn.Linear(in_size, action_dim))
        self.max_action = max_action

        if activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "tanh":
            self.activation_fn == nn.Tanh
        self.steps = 0

    def forward(self, state):
        x = state
        for i in range(len(self.pi)-1):
            x = self.activation_fn(self.pi[i](x))
        y = self.max_action * torch.sigmoid(self.pi[-1](x))
        #if self.steps % 200 == 0:
        #    print(f"state: {state}")
        #    print(f"y: {y}")
        self.steps +=  1
        return y

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.forward(state).cpu().data.numpy().flatten()

    def get_log_prob(self, state):
        '''
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_prob = self.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action_prob[:,action.item()].item()
        '''
        state = torch.FloatTensor(state).to(self.device)
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_probs = self.forward(state)
            #x = action_probs
            #x_normalized = x / x.abs().max() * torch.sign(x)
            #action_probs = (x_normalized +1.) / 2.

            print(f"action_probs shape: {action_probs.shape}")
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        #print(f"logprob: {action_logprob},  {action_logprob.item()}")
        return action_logprob.detach()

    def get_log_prob1(self, state):
        '''
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_prob = self.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action_prob[:,action.item()].item()
        '''
        state = torch.FloatTensor(state).to(self.device)
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_probs = self.forward(state)
            #x = action_probs
            #x_normalized = x / x.abs().max() * torch.sign(x)
            #action_probs = (x_normalized + 1.) / 2.
            #if self.steps % 200 == 0:
            #   print(f"action_probs shape: {action_probs.shape}")
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        #print(f"logprob: {action_logprob},  {action_logprob.item()}")
        dist_entropy = dist.entropy()
        '''
        if self.steps % 1000 == 0:
            print(f"state: {state}")
            print(f"action_probs: {action_probs}")
            print(f"dist: {dist}")
            print(f"action:{action}")
            print(f"action_logprob: {action_logprob}")
            print(f"dist_entropy: {dist_entropy}")
        '''
        return action_logprob.detach(), dist_entropy


class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 qf : list = [400, 300],
                 activation_fn: str = "relu"
                 ):
        super(Critic, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "tanh":
            self.activation_fn == nn.Tanh

        # Q1 architecture
        self.qf1 = nn.Sequential()
        in_size = state_dim + action_dim
        #in_size = state_dim
        for layer_sz in qf:
            self.qf1.append(nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        self.qf1.append(nn.Linear(in_size, 1))

        self.steps = 0

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        #sa = state
        q1 = sa
        for i in range(len(self.qf1)-1):
            q1 = self.activation_fn(self.qf1[i](q1))
        q1 = self.qf1[-1](q1)

        #if self.steps % 200 == 0:
        #    print(f"q1: {q1}")
        self.steps += 1
        return q1

class PPO(object):
    def __init__(
        self,
        max_action,
        hyperparameters,
        train_env,
        device:str = "cpu",
        early_stopping:int = 100_000,
        state_dim:int = 23,  # used when a training environment is not supplied
        action_dim:int = 3,  # used when a training environment is not supplied
        verbose: int = 0
    ):
        self.device=device
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if verbose == 1:
            print(f"Model PPO is configured to device {self.device}")

        self.train_env = train_env

        if self.train_env is not None:
            state_dim = self.train_env.agents[0].get_observation_space_shape()
            action_dim = self.train_env.agents[0].action_space.shape[0]

        self.actor = Actor(state_dim,
                           action_dim, max_action,
                           hyperparameters["net_arch"]["pi"],
                           hyperparameters["activation_fn"],
                           device=device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=hyperparameters["learning_rate"])

        self.critic = Critic(state_dim,
                             action_dim,
                             hyperparameters["net_arch"]["qf"],
                             hyperparameters["activation_fn"]).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=hyperparameters["learning_rate"])

        self.max_action = max_action
        self.buffer_size = hyperparameters["buffer_size"]
        self.discount = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"]
        self.K_epochs = hyperparameters["K_epochs"]
        self.batch_size = hyperparameters["batch_size"]
        self.policy_noise = hyperparameters["policy_noise"] * self.max_action
        self.noise_clip = hyperparameters["noise_clip"] * self.max_action
        self.policy_freq = hyperparameters["policy_freq"]
        self.clip_param = hyperparameters["clip_param"] * self.max_action  # 0.2
        self.max_grad_norm = hyperparameters["max_grad_norm"] * self.max_action  # 0.5
        self.replay_buffer = utils.ReplayMemory(hyperparameters["buffer_size"],
                                                device=self.device)
        self.trackr = tracker.tracker(100)

        # Early stopping
        self.early_stopping = early_stopping
        self.exit = False

        self.ppo_update_time = 5
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, old_state = replay_buffer.sample(self.batch_size)

        R = 0
        rewards1 = []
        reward1 = reward.numpy()
        for r, n in zip(reversed(reward1), reversed(not_done)):
            R = r + (1. - n.numpy()) * self.discount * R
            rewards1.insert(0, R)
        rewards = torch.tensor(np.array(rewards1), dtype=torch.float).to(self.device)
        rewards = rewards.view(-1, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        critic_losses = []
        actor_losses = []

        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

                next_action = (
                self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                next_Q = self.critic_target(next_state, self.actor_target(next_state))
                next_value = rewards + not_done * self.discount * next_Q
                #if self.total_it % 500 == 0:
                #    print(f"reward shape: {reward.shape}, reward: {reward}")
                    #print(f"End target_Q: {target_Q}")

            cur_log_prob, cur_dist_entropy = self.actor.get_log_prob1(state)
            next_log_prob, next_dist_entropy = self.actor_target.get_log_prob1(next_state)
            # Get current Q estimate
            cur_Q = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(cur_Q, next_value)
            if self.total_it % 500 == 0:
                print(f'critic_loss: {critic_loss}')

            advantages = rewards.detach() - cur_Q.detach()
            advtg = advantages.view(-1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(next_log_prob - cur_log_prob)
            # Finding Surrogate Loss
            surr1 = advtg * ratios
            surr2 = advtg * torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param)

            # final loss of clipped objective PPO
            loss1 = -torch.min(surr1, surr2).mean()
            loss1.requires_grad = True

            # Compute actor loss
            loss0 = -self.critic(state, self.actor(state)).mean()

            # act_loss = loss1 + loss3
            if self.total_it % 200 == 0 and i == 0:
                print(
                    f"state, action, reward, not_done shape: {state.shape},{action.shape},{reward.shape},{not_done.shape}")
                print(f"old state shape: {old_state.shape}, ratios shape: {ratios.shape}")
                print(
                    f"advantages shape: {advantages.shape}, advtg shape: {advtg.shape}, rewards shape:{rewards.shape}")
                print(f"log_prob shape: {cur_log_prob.shape}, dist_entropys shape: {cur_dist_entropy.shape}")
                print(f"cur_Q shape: {cur_Q.shape}, ratios shape:{ratios.shape}")
                print(f"critic_loss: {critic_loss}, loss 0:{loss0} , loss1:{loss1} ")
                print(f"surr1 shape: {surr1.shape}, surr2 shape: {surr2.shape}")

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()



            if self.total_it % 500 == 0:
                print(f'loss1: {loss1}')

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            loss1.backward()
            self.actor_optimizer.step()

            actor_losses.append(loss1.detach().numpy())
            critic_losses.append(critic_loss.detach().numpy())

        if self.total_it % 1 == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        # Update the frozen target models
        #for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        #for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        critic_loss = np.mean(critic_losses)
        actor_loss = np.mean(actor_losses)
        if self.total_it % 200 == 0:
            print(f"critic_loss: {critic_loss}, actor_loss: {actor_loss}")
        return critic_loss, actor_loss


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def explore_for_expert_targets(self,
                                   reward_target_exploration_steps=25_000):
        if self.train_env is None:
            print("Model cannot explore because training envrionment is\
                  missing. Please reload model and supply a training envrionment.")
            return

        self.done = False
        for t in range(reward_target_exploration_steps):
            obs_vec = self.train_env.step(self.actor, random=True)

            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True

            if self.done:
                self.train_env.reset()
                self.done = False
            	#env.tracker.create_video()
                self.train_env.tracker.reset()

        self.train_env.reset()
        self.done = False

    def learn(self,
              timesteps,
              callback,
              start_timesteps=25_000,
              incremental_replay_buffer = None):

        if self.train_env is None:
            print("Model cannot explore because training envrionment is\
                  missing. Please reload model and supply a training envrionment.")
            return

        next_update_at = self.buffer_size*2

        episode_reward = 0
        episode_timesteps = 0
        self.episode_num = 0

        callback.on_training_start()

        self.train_env.reset()
        self.done = False
        start_time = time.clock_gettime(time.CLOCK_REALTIME)

        episode_start_time = start_time

        for t in range(1,int(timesteps)+1):
            self.num_timesteps = t

            episode_timesteps += 1
            if t < start_timesteps:
                obs_vec = self.train_env.step(model=self.actor, random=True,  rl_model_type = 'PPO')
            else:
                obs_vec = self.train_env.step(model=self.actor, random=False,  rl_model_type = 'PPO')

            all_rewards = []
            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True
                all_rewards.append(indiv_obs[2])
                transition = (indiv_obs[0], indiv_obs[3], indiv_obs[1], indiv_obs[2], 1. -indiv_obs[4], indiv_obs[5])
                self.replay_buffer.add(*transition)

            episode_reward += float(np.mean(np.array(all_rewards)))

            if t >= start_timesteps:
                critic_loss, actor_loss = self.train(self.replay_buffer)

            if self.done:
                episode_finish_time = time.clock_gettime(time.CLOCK_REALTIME)
                if t < start_timesteps:
                    self.trackr.append(actor_loss=0,
                                       critic_loss=0,
                                       episode_reward=episode_reward,
                                       episode_length = episode_timesteps,
                                       episode_fps = episode_timesteps / (episode_finish_time - episode_start_time))
                else:
                    self.trackr.append(actor_loss=actor_loss,
                           critic_loss=critic_loss,
                           episode_reward=episode_reward,
                           episode_length = episode_timesteps,
                           episode_fps = episode_timesteps / (episode_finish_time - episode_start_time))

            callback.on_step()
            if self.done:
                self.train_env.reset()
                self.done = False
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1
                self.train_env.tracker.reset()
                episode_start_time = time.clock_gettime(time.CLOCK_REALTIME)

            # Early stopping
            if self.exit is True:
                print(f"Early stopping mechanism triggered at timestep=\
                      {self.num_timesteps} after {self.early_stopping} steps\
                       without improvement ... Learning terminated.")
                break

            if incremental_replay_buffer is not None:
                if t >= next_update_at:
                    if incremental_replay_buffer == "double":
                        self.buffer_size *= 2
                        next_update_at += self.buffer_size * 2
                    elif incremental_replay_buffer == "triple":
                        self.buffer_size *= 3
                        next_update_at += self.buffer_size# * 3
                    elif incremental_replay_buffer == "quadruple":
                        self.buffer_size *= 4
                        next_update_at += self.buffer_size# * 3

                    old_replay_buffer = self.replay_buffer
                    self.replay_buffer = utils.ReplayMemory(self.buffer_size,
                                                            device=self.device)
                    self.replay_buffer.add_content_of(old_replay_buffer)

                    print(f"Updated replay buffer at timestep {t};\
                           replay_buffer_size={self.buffer_size},\
                           len={self.replay_buffer.__len__()}\
                           next_update_at={next_update_at}")

        callback.on_training_end()
