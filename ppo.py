import math
import random
from collections import namedtuple, deque

import gym
import numpy as np
import gym_minigrid

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from IPython.display import clear_output
import matplotlib.pyplot as plt

from misc_utils import one_hot, Transition, ReplayMemory
from common.multiprocessing_env import SubprocVecEnv

# Hyper Parameters
MINI_BATCH_SIZE  = 256
LR = 1e-3                   # learning rate
GAMMA = 0.99                # reward discount
TAU = 0.95                  # average parameter
PPO_EPOCHS = 4 
MAX_FRAMES = 1e6
T_STEPS = 1200
ENV_NAME = "MiniGrid-DoorKey-5x5-v0"

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def init_params(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # image network
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, num_outputs)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(init_params)

    def forward(self, image, data):
        image = image.transpose(1, 3).transpose(2, 3)
        x1 = self.image(image)
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = data
        x = torch.cat((x1, x2), dim = 1)

        dist = self.actor(x)
        dist = Categorical(logits=F.log_softmax(dist, dim=1))

        value = self.critic(x)
        return dist, value

# calculate GAE advantage 
def compute_gae(next_value, rewards, masks, values, gamma = 0.99, tau = 0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        # print(f'rewards:{rewards[step]}\tvalues:{[step + 1]}\tmask:{masks[step]}')
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# iterate mini_batch used for optimization
def ppo_iter(mini_batch_size, states_image, states_data, actions, log_probs, returns, advantage):
    batch_size = states_image.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states_image[rand_ids, :], states_data[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

# update
def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states_image, states_data, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for states_image, states_data, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states_image, states_data, actions, log_probs, returns, advantages):
            dist, value = model(states_image, states_data)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def state_process(state, goal_pos, current_pos):
    direction = state['direction']
    s = np.append(direction, goal_pos)
    s = np.append(s, current_pos)
    image = state['image']
    state = {"image": torch.Tensor(image).unsqueeze(0), "data": torch.Tensor(s).unsqueeze(0)}
    return state

#####################################
# num_envs = 8

# def make_env():
#     def _thunk():
#         env = gym.make(ENV_NAME)
#         return env

#     return _thunk

# envs = [make_env() for i in range(num_envs)]
# envs = SubprocVecEnv(envs)
env  = gym.make(ENV_NAME)
env.seed(0)
n = env.observation_space["image"].shape[0]
m = env.observation_space["image"].shape[1]
image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
embedding_size = image_embedding_size + 2 + 2 + 1
act_shape = env.action_space.n

# GET GOAL POSE
for grid in env.grid.grid:
    if grid is not None and grid.type == "goal":
        goal_pose = grid.cur_pos

model = ActorCritic(embedding_size, act_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

state = env.reset()
frame_idx = 0
ep_reward = 0
avg_rw = deque(maxlen=30)
ep_number = 0

while frame_idx < MAX_FRAMES:
    # variables to record experience
    log_probs = []
    values = []
    states_image = []
    states_data = []
    actions = []
    rewards = []
    masks = []
    entropy = 0

    # collect experience
    for _ in range(T_STEPS):
        state_processed = state_process(state, goal_pose, env.agent_pos)
        dist, value = model(state_processed["image"], state_processed["data"])

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob.unsqueeze(1))
        values.append(value)
        rewards.append(torch.Tensor([reward]).unsqueeze(1))
        masks.append(torch.Tensor([1-done]).unsqueeze(1))

        states_image.append(state_processed["image"])
        states_data.append(state_processed["data"])
        actions.append(action.unsqueeze(1))
        state = next_state
        frame_idx += 1

        ep_reward += reward
        if done:
            state = env.reset()
            ep_number += 1
            avg_rw.append(ep_reward)
            ep_reward = 0
            done = False
        
        if frame_idx % 2000 == 0:
            print(f'frames:{frame_idx}\teps:{ep_number}\tavg_rw:{np.mean(avg_rw)}')

    next_state_processed = state_process(next_state, goal_pose, env.agent_pos)
    _, next_value = model(next_state_processed["image"], next_state_processed["data"])
    returns = compute_gae(next_value, rewards, masks, values, GAMMA, TAU)
    # print(len(returns))
    # print(len(values))
    # print(returns)
    # print(values)
    # print(torch.cat(returns).size())
    # print(torch.cat(values).size())
    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states_image    = torch.cat(states_image)
    states_data    = torch.cat(states_data)
    actions   = torch.cat(actions)
    advantages = returns - values

    ppo_update(model, optimizer, PPO_EPOCHS, MINI_BATCH_SIZE, states_image, states_data, actions, log_probs, returns, advantages)




