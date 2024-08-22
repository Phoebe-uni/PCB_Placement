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
            self.activation_fn = nn.Tanh
        elif activation_fn == "softmax":
            self.activation_fn = F.softmax

        self.steps = 0


    def forward(self, state):
        x = state
    #    if self.steps % 1000 == 0:
    #        print(f"state: {x}")

        for i in range(len(self.pi)-1):
            x = self.activation_fn(self.pi[i](x))
            #if self.steps % 1000 == 0:
            #    print(f"i: {i}, x: {x}")

        y = self.max_action * F.softmax(self.pi[-1](x), dim = -1)
        #y = self.max_action * F.relu(self.pi[-1](x), inplace=True)
        #if self.steps % 1000 == 0:
        #    print(f"pi[-1]x: {self.pi[-1](x)}")
        #    print(f"y Shape: {y.shape}, y: {y}")
        self.steps += 1
        return y

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
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
        action_probs = self.forward(state)
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
        action_probs = self.forward(state)
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
            self.activation_fn = nn.Tanh
        elif activation_fn == "softmax":
            self.activation_fn = F.softmax
        self.steps = 0

    def forward(self, state):
        x = state
        #if self.steps % 1000 == 0:
        #    print(f"state: {x}")

        for i in range(len(self.pi)):
            x = self.activation_fn(self.pi[i](x))
            #if self.steps % 1000 == 0:
            #    print(f"i Layer: {i}, x: {x}")

        value = self.state_value(x)
        #if self.steps % 1000 == 0:
        #    print(f"value: {value}")
        self.steps += 1
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
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': hyperparameters["learning_rate"]},
            {'params': self.critic.parameters(), 'lr': hyperparameters["lr_critic"]}
        ])
        self.max_action = max_action
        self.buffer_size = hyperparameters["buffer_size"]
        self.discount = hyperparameters["gamma"]
        self.K_epochs = hyperparameters["K_epochs"]
        self.batch_size = hyperparameters["batch_size"]
        #self.policy_noise = hyperparameters["policy_noise"] * self.max_action       #0.2
        #self.noise_clip = hyperparameters["noise_clip"] * self.max_action           #0.5

        self.clip_param = hyperparameters["clip_param"] * self.max_action            #0.2
        self.max_grad_norm = hyperparameters["max_grad_norm"] * self.max_action      #0.5
        self.tau = hyperparameters["tau"]
        self.policy_freq = hyperparameters["policy_freq"]

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
        self.MseLoss = nn.MSELoss()


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

    def train(self, replay_buffer):

        self.total_it += 1
        state, action, next_state, reward,  not_done, old_state = replay_buffer.sample(self.batch_size)

        if self.training_step % 500 == 0:
            print(f"state: {state}")
            print(f"next_state: {next_state}")
            print(f"old_state: {old_state}")
            print(f"reward: {reward}")
            print(f"action: {action}")
            print(f"not_done: {not_done}")

        R = 0
        Gt = []
        reward1 = reward.numpy()
        #for r in reward1[::-1]:
        #    R = r + self.discount * R
        #    Gt.insert(0, R)
        for r, n in  zip(reversed(reward1), reversed(not_done)) :
            R = r + n.numpy() * self.discount * R
            Gt.insert(0, R)

        Gt = torch.tensor(np.array(Gt), dtype=torch.float).to(self.device)


        #print("The agent is updateing....")



        Gt_index = Gt.view(-1, 1)
        if self.training_step % 500 == 0:
            print(f'reward1: {reward1}')
            print(f"Gt: {Gt}")
            print(f"Gt_index : {Gt_index}")
        Gt_index = (Gt_index - Gt_index.mean()) / (Gt_index.std() + 1e-7)



        OldV = self.critic(state)
        V = self.critic(next_state)
        delta = Gt_index - OldV
        advantage = delta


        if self.training_step % 500 == 0:
            print(f"Gt_index : {Gt_index}")
            print(f"OldV: {OldV}")
            print(f"V: {V}")
            print(f"delta: {delta}")

        old_log_prob, old_dist_entropy = self.actor.get_log_prob1(state)
        log_prob, dist_entropy = self.actor.get_log_prob1(next_state)

        ratio = torch.exp(log_prob - old_log_prob)
        #print(f"ratio: {ratio}")
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

        loss1 = torch.min(surr1, surr2)
        loss2 = 0.5*self.MseLoss(V, Gt_index)
        loss3 = - 0.01 * dist_entropy
        loss = -loss1 + loss2 + loss3
        #loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(V, Gt_index) - 0.01 * dist_entropy

        loss_mean = loss.mean()

        if self.training_step % 500 == 0:
            print('losses {}'.format(loss))
            print(f"advantage: {advantage}")

            print(f"ratio: {ratio}")
            print(f"surr1: {surr1}")
            print(f"surr2: {surr2}")
            print(f"old_log_prob: {old_log_prob}")
            print(f"log_prob: {log_prob}")
            print(f"dist_entropy: {dist_entropy}")
            print(f'loss: {loss_mean}, loss1 : {loss1.mean()}, loss2 : {loss2.mean()}, loss3: {loss3.mean()}')

        # take gradient stepf
        self.optimizer.zero_grad()
        #loss_mean.backward()
        loss.mean().backward(retain_graph=True)
        loss2.mean().backward(retain_graph=True)
        ''' # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()
                                           ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()
                                           ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        '''
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.training_step += 1
        loss1 = loss_mean.detach().numpy()
        return loss1, loss1

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
                obs_vec = self.train_env.step(model=self.actor, random=True, rl_model_type = 'PPO')
            else:
                obs_vec = self.train_env.step(model=self.actor, random=False,  rl_model_type = 'PPO')

            all_rewards = []
            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True
                all_rewards.append(indiv_obs[2])
                transition = (indiv_obs[0], indiv_obs[3], indiv_obs[1], indiv_obs[2], 1. - indiv_obs[4], indiv_obs[5])
                # Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
                # PPOTransition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
                #print(f"trainsition: {transition}")
                self.replay_buffer.add(*transition)

                #action, action_prob = self.select_action(indiv_obs[0])
                #trans = PPOTransition(indiv_obs[0], action, action_prob, indiv_obs[2], indiv_obs[1])
                #self.store_transition(trans)

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
