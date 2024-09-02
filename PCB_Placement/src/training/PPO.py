import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.distributions import MultivariateNormal
import math

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
        self.action_dim = action_dim
        self.state_dim = state_dim
        in_size = state_dim
        for layer_sz in pi:
            self.pi.append(nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        #self.pi.append(nn.Linear(in_size, action_dim))
        self.max_action = max_action

        self.action_mean = nn.Linear(in_size, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.log_std = 0.0
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * self.log_std)

        if activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "tanh":
            self.activation_fn = torch.tanh
        elif activation_fn == 'sigmoid':
            self.activation_fn = torch.sigmoid



        self.steps = 0

    def forward(self, state):
        x = state
        for i in range(len(self.pi)):
            x = self.activation_fn(self.pi[i](x))
        action_mean = self.action_mean(x)
        #print(f"action_log_std shape:{self.action_log_std.shape}")
        #print(f"x shape: {x.shape}, action_mean :{action_mean.shape}")
        if len(action_mean.shape) == 2 :
            action_log_std = nn.Parameter(torch.ones(1, self.action_dim) * self.log_std)
        elif len(action_mean.shape) == 1:
            action_log_std = nn.Parameter(torch.ones(self.action_dim) * self.log_std)
        else:
            action_log_std = nn.Parameter(torch.ones(self.action_dim) * self.log_std)
        action_log_std1 = action_log_std.expand_as(action_mean)


        action_std = torch.exp(action_log_std1)
        #if self.steps % 200 == 0:
        #    print(f"state: {state}")
        #    print(f"y: {y}")
        self.steps +=  1
        return action_mean, action_log_std, action_std

    def select_action(self, state):
        #x = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        x = torch.FloatTensor(state).to(self.device)
        action_mean, _, action_std = self.forward(x)

        action2 = torch.normal(action_mean.to(self.device), action_std.to(self.device))
        #action = torch.tanh(action2)
        action = action2/torch.sqrt(torch.abs(action_mean))

        action = torch.where(torch.isnan(action), torch.full_like(action, 0), action)
        action = torch.where(torch.isinf(action), torch.full_like(action, 0), action)
        noise = np.random.normal(0,0.1, size=3)
        #if self.steps % 10000 == 0:
        #    print(f"action mean: {action_mean}, std: {action_std}")
        #    print(f"action nomal: {action2},  --clip: {action}")
        #    print(f"noise: {noise}")
        #action1 = action.cpu().detach().numpy().flatten()
        action1 = action.cpu().detach().numpy()
        #print(f"Shape: state: {state.shape}, action: {action.shape}, action1: {action1.shape}")
        return action1
    def select_action1(self, state):
        #x = torch.FloatTensor(state).to(self.device)
        x = state
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean.to(self.device), action_std.to(self.device))
        action1 = action
        #print(f"Shape: state: {state.shape}, action: {action.shape}, action1: {action1.shape}")
        return action1

    def get_log_prob(self, x, actions):
        actions = actions.to(self.device)
        action_mean, action_log_std, action_std = self.forward(x)
        var = action_std.pow(2)
        val11 = -(actions - action_mean).pow(2)
        val1 = val11.cpu() / (2. * var.cpu())
        val2 = - 0.5 * math.log(2 * math.pi)
        val3= val1 - action_log_std.cpu()
        log_density = val3 + val2
        log_density1 = log_density.sum(1, keepdim=True)
        return log_density1.to(self.device)


    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def act(self, state):
        action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def get_prob(self, state, action):
        action_mean = self.forward(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


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
            self.activation_fn = torch.tanh
        elif activation_fn == 'sigmoid':
            self.activation_fn = torch.sigmoid

        # Q1 architecture
        self.qf1 = nn.Sequential()
        #in_size = state_dim + action_dim
        in_size = state_dim
        for layer_sz in qf:
            self.qf1.append(nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        #self.qf1.append(nn.Linear(in_size, 1))
        self.value_head = nn.Linear(in_size, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.steps = 0

    def forward(self, state, action):
        #sa = torch.cat([state, action], 1)
        sa = state
        q1 = sa
        for i in range(len(self.qf1)):
            q1 = self.activation_fn(self.qf1[i](q1))
        value = self.value_head(q1)

        #if self.steps % 200 == 0:
        #    print(f"q1: {q1}")
        self.steps += 1
        return value

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

        self.l2_reg = 1e-3
        self.total_it = 0
        self.loss = nn.MSELoss()


    def train(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, old_state = replay_buffer.sample(self.batch_size)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip)

        old_action = self.actor.select_action1(old_state)

        old_action = old_action + noise

        #R = 0
        #reward1 = reward.cpu().numpy()

        #for r, n in zip(reversed(reward1), reversed(not_done)):
        #    R = r + (1. - n.cpu().numpy()) * self.discount  * R
        #    rewards1.insert(0, R)
        #rewards = torch.tensor(np.array(rewards1), dtype=torch.float).to(self.device)

        values = self.critic(state, action)

        tensor_type = type(reward)
        deltas = tensor_type(reward.size(0), 1).to(self.device)
        advantages = tensor_type(reward.size(0), 1).to(self.device)
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(reward.size(0))):
            deltas[i] = reward[i] + self.discount * prev_value * not_done[i] - values[i]
            advantages[i] = deltas[i] + self.discount * self.tau * prev_advantage * not_done[i]

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]
        returns = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        critic_losses = []
        actor_losses = []

        #old_action = self.actor.select_action1(old_state)
        fixed_log_probs = self.actor.get_log_prob(old_state, old_action)

        #if self.total_it % 1000 == 0 :
        #    print(
        #        f"state, action, reward, not_done shape: {state.shape},{action.shape},{reward.shape},{not_done.shape}")
        #    # print(f"not_done: {not_done}")
        #    print(f"Shape: Advantages: {advantages.shape}, returns: {returns.shape}")
        #    print(f"shape: old_action: {old_action.shape} , fixed_log_probs: {fixed_log_probs.shape}")

        for i in range(self.K_epochs):
            values_pred = self.critic(state, action)
            value_loss = (values_pred - returns).pow(2).mean()
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg

        #    if self.total_it % 1000 == 0 and i == 0:
        #        print(f"Shape: value_pred:{values_pred.shape}")
        #        print(f"value_loss: {value_loss}")

            critic_losses.append(value_loss.detach().cpu().numpy())
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)


        log_probs = self.actor.get_log_prob(state, action)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        #policy_surr.requires_grad_ = True

        #if self.total_it % 1000 == 0:
        #    print(f"Shape: ratio:{ratio.shape}, surr1: {surr1.shape}, surr2: {surr2.shape}")
        #    print(f"policy_surr: {policy_surr}")
        actor_losses.append(policy_surr.detach().cpu().numpy())

        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.critic_optimizer.step()
        self.actor_optimizer.step()


        '''
        old_states = state
        old_actions, old_logprobs = self.actor_target.act(state)
        old_state_values = self.critic_target(state, old_actions)

        advantages = rewards.detach() - old_state_values.detach()
        advtg = advantages.view(-1)

        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            with torch.no_grad():

                state_values = self.critic(old_states, old_actions)
                logprobs, dist_entropy = self.actor.get_prob(old_states, old_actions)
                state_values = self.critic(state, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())


            # Finding Surrogate Loss
            surr1 = ratios * advtg
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advtg

            # final loss of clipped objective PPO
            loss1 = - torch.min(surr1, surr2).mean()
            loss1.requires_grad = True
            loss2 = 0.5 * self.loss(state_values, rewards)
            loss2.requires_grad = True
            loss3 = (- 0.01 * dist_entropy).mean()
            loss = loss1 + loss2 + loss3


            loss0 = -self.critic(state, self.actor(state)).mean()


            # act_loss = loss1 + loss3
            if self.total_it % 1000 == 0 and i == 0:
                print(
                    f"state, action, reward, not_done shape: {state.shape},{action.shape},{reward.shape},{not_done.shape}")
                #print(f"not_done: {not_done}")
                print(f"old state shape: {old_state.shape}, ratios shape: {ratios.shape}")
                print(
                    f"advantages shape: {advantages.shape}, advtg shape: {advtg.shape}, rewards shape:{rewards.shape}")
                print(f"log_prob shape: {old_logprobs.shape}, dist_entropys shape: {dist_entropy.shape}")
                print(f"state_values shape: {state_values.shape}, ratios shape:{ratios.shape}")
                print(f"loss0: {loss0}, loss: {loss}, loss1: {loss1}, loss 2:{loss2} , loss3:{loss3} ")
                print(f"surr1 shape: {surr1.shape}, surr2 shape: {surr2.shape}")

                # take gradient stepf

                # Optimize the critic
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            loss0.backward()
            loss.backward()
            #loss2.backward()
            #loss1.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            #if self.total_it % 1000 == 0 and i == 0:
            #    print(f'loss1: {loss1}, loss2: {loss2}, loss3: {loss3}')

            # Optimize the actor


            actor_losses.append(loss1.detach().numpy())
            critic_losses.append(loss2.detach().numpy())
        '''
        #if self.total_it % 2 == 0:
        #    self.actor_target.load_state_dict(self.actor.state_dict())
        #    self.critic_target.load_state_dict(self.critic.state_dict())
        # Update the frozen target models
        #for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        #for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        critic_loss = np.mean(critic_losses)
        actor_loss = np.mean(actor_losses)
        if self.total_it % 5000 == 0:
            print(f"PPO {self.total_it},  critic_loss: {critic_loss}, actor_loss: {actor_loss}")
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

        for t in range(1,int(timesteps)):
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
                    actor_loss = 0
                    critic_loss = 0

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

                if t%5000 == 0:
                    print(f"PPO {t}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, ep_reward: {episode_reward}")
                    print(f"PPO {t}, epi_length: {episode_timesteps}, epi_fps: {episode_timesteps / (episode_finish_time - episode_start_time)}")

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
