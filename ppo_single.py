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

# Hyper Parameters
MINI_BATCH_SIZE  = 256
LR = 5e-4                   # learning rate
GAMMA = 0.99                # reward discount
TAU = 0.95                  # average parameter
PPO_EPOCHS = 4 
MAX_FRAMES = 1e6
T_STEPS = 128
MAX_GRAD_NORM = 0.5
ENV_NAME = "MiniGrid-Empty-5x5-v0"

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
batch_num = 0

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

        value = self.critic(x).squeeze(1)
        return dist, value

# calculate GAE advantage 
def compute_gae(next_value, rewards, masks, values, gamma = 0.99, tau = 0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        # print(f'rewards:{rewards[step]}\tvalues:{[step + 1]}\tmask:{masks[step]}')
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages

# iterate mini_batch used for optimization
def get_ppo_batch_index(recurrence = 1):
    global batch_num
    batch_size = states_image.size(0)
    indexes = np.arange(0, batch_size, recurrence)
    indexes = np.random.permutation(indexes)
    # Shift starting indexes by self.recurrence//2 half the time
    if batch_num % 2 == 1:
        indexes = indexes[(indexes + recurrence) % T_STEPS != 0]
        indexes += recurrence // 2
    batch_num += 1

    num_indexes = MINI_BATCH_SIZE // recurrence
    batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

    return batches_starting_indexes

# update
def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states_image, states_data, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for indexs in get_ppo_batch_index():
            states_image_ = states_image[indexs]
            states_data_ = states_data[indexs]
            action = actions[indexs]
            old_log_probs = log_probs[indexs]
            return_ = returns[indexs]
            advantage = advantages[indexs]

            dist, value = model(states_image_, states_data_)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()

            value_clipped = return_ - advantage + torch.clamp(value - (return_ - advantage), -clip_param, clip_param)
            surr1 = (value - return_).pow(2)
            surr2 = (value_clipped - return_).pow(2)
            critic_loss = torch.max(surr1, surr2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

def state_process(state, goal_pos, current_pos):
    direction = state['direction']
    s = np.append(direction, goal_pos)
    s = np.append(s, current_pos)
    image = state['image']
    return torch.Tensor(image).unsqueeze(0), torch.Tensor(s).unsqueeze(0)

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
        image, data = state_process(state, goal_pose, env.agent_pos)
        with torch.no_grad():
            dist, value = model(image, data)

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.Tensor([reward]))
        masks.append(1 - torch.Tensor([done]))

        states_image.append(image)
        states_data.append(data)
        actions.append(action)
        state = next_state
        frame_idx += 1

        ep_reward += reward
        if done:
            state = env.reset()
            for grid in env.grid.grid:
                if grid is not None and grid.type == "goal":
                    goal_pose = grid.cur_pos
            ep_number += 1
            avg_rw.append(ep_reward)
            ep_reward = 0
            done = False
        
        if frame_idx % 2000 == 0:
            print(f'frames:{frame_idx}\teps:{ep_number}\tavg_rw:{np.mean(avg_rw)}')

    with torch.no_grad():
        next_image, next_data = state_process(next_state, goal_pose, env.agent_pos)
        _, next_value = model(next_image, next_data)
    returns, advantages = compute_gae(next_value, rewards, masks, values, GAMMA, TAU)

    rewards   = torch.cat(rewards).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states_image    = torch.cat(states_image)
    states_data    = torch.cat(states_data)
    actions   = torch.cat(actions)
    advantages   = torch.cat(advantages)
    returns   = torch.cat(returns).detach()

    # print('---')
    # print(returnn.shape)
    # print(log_probs.shape)
    # print(values.shape)
    # print(returnn.shape)
    # print(advantages.shape)
    # print(actions.shape)
    # print('---')

    ppo_update(model, optimizer, PPO_EPOCHS, MINI_BATCH_SIZE, states_image, states_data, actions, log_probs, returns, advantages)




