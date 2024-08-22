import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import tracker
import time

#Implementation of Double Q-learning Network (DQN) with pytorch
#riginal paper:
#Written by YiFei, Yu


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 pi : list = [400, 300],
                 activation_fn: str = "relu",
                 device: str = "cuda",
                 epsilon = 0.9
                 ):

        super(ActorNet, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.epsilon = epsilon
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
        y = self.max_action * torch.tanh(self.pi[-1](x))
        #if self.steps % 200 == 0:
        #    print(f"state: {state}")
        #    print(f"y: {y}")
        self.steps +=  1
        return y

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.forward(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= self.epsilon: # epslion greedy
            action = np.random.choice(range(self.action_dim), 1).item()
        return action

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
        for layer_sz in qf:
            self.qf1.append(nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        self.qf1.append(nn.Linear(in_size, 1))

        self.steps = 0

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = sa
        for i in range(len(self.qf1)-1):
            q1 = self.activation_fn(self.qf1[i](q1))
        q1 = self.qf1[-1](q1)

        #if self.steps % 200 == 0:
        #    print(f"q1: {q1}")
        self.steps += 1
        return q1


class DQN(object):
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
            print(f"Model DQN is configured to device {self.device}")

        self.train_env = train_env

        if self.train_env is not None:
            self.state_dim = self.train_env.agents[0].get_observation_space_shape()
            self.action_dim = self.train_env.agents[0].action_space.shape[0]

        self.epsilon = hyperparameters["epsilon"]

        self.actor = ActorNet(self.state_dim,
                           self.action_dim, max_action,
                           hyperparameters["net_arch"]["pi"],
                           hyperparameters["activation_fn"],
                           device=device, epsilon=self.epsilon).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=hyperparameters["learning_rate"])
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim,
                             action_dim,
                             hyperparameters["net_arch"]["qf"],
                             hyperparameters["activation_fn"]).to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=hyperparameters["lr_critic"])

        self.critic_target = copy.deepcopy(self.critic)
        self.max_action = max_action
        self.buffer_size = hyperparameters["buffer_size"]
        self.discount = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"]
        self.batch_size = hyperparameters["batch_size"]
        self.policy_noise = hyperparameters["policy_noise"] * self.max_action
        self.noise_clip = hyperparameters["noise_clip"] * self.max_action
        self.policy_freq = hyperparameters["policy_freq"]


        self.replay_buffer = utils.ReplayMemory(hyperparameters["buffer_size"],
                                                device=self.device)
        self.trackr = tracker.tracker(100)

        # Early stopping
        self.early_stopping = early_stopping
        self.exit = False
        self.MseLoss = nn.MSELoss()

        self.total_it = 0

    def select_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        #return self.actor(state).cpu().data.numpy().flatten()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.actor(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= self.epsilon: # epslion greedy
            action = np.random.choice(range(self.action_dim), 1).item()
        return action

    def train(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, old_state = replay_buffer.sample(self.batch_size)
        #action1 = action.long()
        #if self.total_it % 200 == 1:
        #    print(f"state shape: {state.shape}, state:{state}")
        #    print(f"action shape: {action.shape}, action:{action}")
        #    print(f"reward shape: {reward.shape}, reward:{reward}")

        if self.total_it % 300 == 1:
            #print(f"reward shape: {reward.shape}, rewards: {reward}")
            print(f"not_done shape: {not_done.shape}, not_done: {not_done}")

        R = 0
        rewards1 = []
        reward1 = reward.numpy()
        for r, n in zip(reversed(reward1), reversed(not_done)):
            R = r + n.numpy() * self.discount * R
            rewards1.insert(0, R)
        rewards = torch.tensor(np.array(rewards1), dtype=torch.float).to(self.device)

        rewards = rewards.view(-1, 1)

        v = self.critic(state, action)

        with torch.no_grad():
            v_target = self.critic_target(next_state, self.actor_target(next_state))
            v_target = reward + self.discount * not_done * v_target
            #v_next_target = self.critic_target(next_state, self.actor_target(next_state))
            #v_target = reward + self.discount * not_done * v_next_target

            #v_target = rewards + self.discount * not_done * \
            #                  v_next_target.max(1)[0].view(v.size(0), 1)
            #target_v = reward + self.discount * self.actor_target(next_state).max(1)[0]

        if self.total_it % 300 == 1:
            #print(f"rewards shape: {rewards.shape}, rewards: {rewards}")
            print(f"v_target shape: {v_target.shape}")
            #print(f"v_next_target shape: {v_next_target.shape}")
            print(f"v shape: {v.shape}")
        loss = self.MseLoss(v_target, v)
        if self.total_it % 300 == 1:
            print(f"loss: {loss}")


        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        if self.total_it % 300 == 0:
            print(f'actor_loss: {actor_loss}')

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # if self.total_it % 1 == 0:
        #    self.actor_target.load_state_dict(self.actor.state_dict())

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.cpu().detach().numpy(), loss.cpu().detach().numpy()


    def save(self, filename):

        #torch.save(self.critic.state_dict(), filename + "_critic")
        #torch.save(self.critic_optimizer.state_dict(),
        #           filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        #self.critic.load_state_dict(torch.load(filename + "_critic"))
        #self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        #self.critic_target = copy.deepcopy(self.critic)

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
                obs_vec = self.train_env.step(model=self.actor, random=True,  rl_model_type = 'DQN')
            else:
                obs_vec = self.train_env.step(model=self.actor, random=False,  rl_model_type = 'DQN')

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
