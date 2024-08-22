import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple

import utils
import tracker
import time

#Implement PPO
#Writen by YYF.

PPOTransition = namedtuple('Transition', ('state', 'action',  'a_log_prob', 'reward', 'next_state'))

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
        elif activation_fn == "softmax":
            self.activation_fn == F.softmax


    def forward(self, state):
        x = state
        for i in range(len(self.pi)-1):
            x = self.activation_fn(self.pi[i](x))
        return self.max_action * F.softmax(self.pi[-1](x), dim = -1)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.forward(state).cpu().data.numpy().flatten()

class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 pi : list = [400, 300],
                 activation_fn: str = "relu",
                 device: str = "cuda"
                 ):

        super(Critic, self).__init__()
        self.device = device
        self.pi = nn.Sequential()
        in_size = state_dim
        for layer_sz in pi:
            self.pi.append(nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        self.pi.append(nn.Linear(in_size, action_dim))
        self.state_value = nn.Linear(action_dim, 1)

        if activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "tanh":
            self.activation_fn == nn.Tanh
        elif activation_fn == "softmax":
            self.activation_fn == F.softmax

    def forward(self, state):
        x = state
        for i in range(len(self.pi)):
            x = self.activation_fn(self.pi[i](x))
        value = self.state_value(x)
        return value

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.forward(state).cpu().data.numpy().flatten()

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
        super(PPO, self).__init__()
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
                                                 lr=hyperparameters["lr_critic"])

        self.max_action = max_action
        self.buffer_size = hyperparameters["buffer_size"]
        self.discount = hyperparameters["gamma"]
        #self.tau = hyperparameters["tau"]
        self.batch_size = hyperparameters["batch_size"]
        #self.policy_noise = hyperparameters["policy_noise"] * self.max_action       #0.2
        #self.noise_clip = hyperparameters["noise_clip"] * self.max_action           #0.5

        self.clip_param = hyperparameters["clip_param"] * self.max_action            #0.2
        self.max_grad_norm = hyperparameters["max_grad_norm"] * self.max_action      #0.5

        #self.policy_freq = hyperparameters["policy_freq"]

        self.replay_buffer = utils.ReplayMemory(hyperparameters["buffer_size"],
                                                device=self.device)
        self.trackr = tracker.tracker(100)
        # Early stopping
        self.early_stopping = early_stopping
        self.exit = False
        self.total_it = 0

        self.training_step = 0
        self.ppo_update_time = 5
        self.counter = 0
        self.buffer = []


    def select_action(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def train(self, i_ep):

        self.total_it += 1

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.discount * R
            Gt.insert(0, R)
        Gt = torch.tensor(np.array(Gt), dtype=torch.float)
        #print("The agent is updateing....")

        critic_losses = []
        actor_losses = []

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                actor_losses.append(action_loss.detach())
                #self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                if self.training_step % 1000 ==0:
                    print('action_loss {}'.format(action_loss))
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                critic_losses.append(value_loss.detach())
                if self.training_step % 1000 ==0:
                    print('value_loss {}'.format(value_loss))

                #self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_optimizer.zero_grad()
                value_loss.backward()

                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

        del self.buffer[:] # clear experience
        actor_loss = float(np.mean(actor_losses))
        critic_loss = float(np.mean(critic_losses))
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
                obs_vec = self.train_env.step(model=self.actor, random=True)
            else:
                obs_vec = self.train_env.step(model=self.actor, random=False)

            all_rewards = []
            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True
                all_rewards.append(indiv_obs[2])
                transition = (indiv_obs[0], indiv_obs[3], indiv_obs[1], indiv_obs[2], 1. -indiv_obs[4])
                # Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
                # PPOTransition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
                self.replay_buffer.add(*transition)
                action, action_prob = self.select_action(indiv_obs[0])
                trans = PPOTransition(indiv_obs[0], action, action_prob, indiv_obs[2], indiv_obs[1])
                self.store_transition(trans)

            episode_reward += float(np.mean(np.array(all_rewards)))

            if t >= start_timesteps:
                if self.done:
                    if len(self.buffer) >= self.batch_size:
                        critic_loss, actor_loss = self.train(t)


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
