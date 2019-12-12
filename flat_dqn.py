import math
import random
from collections import namedtuple, deque

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

from IPython.display import clear_output
import matplotlib.pyplot as plt

from misc_utils import one_hot, Transition, ReplayMemory

# Hyper Parameters
BATCH_SIZE = 128
LR = 1e-5                   # learning rate
GAMMA = 0.99                 # reward discount
MEMORY_CAPACITY = 8000
MAX_EPISODES = 3000
MAX_STEPS = int(1e5)
UPDATE_EVERY = int(100)
TRAIN_EVERY = 4
ENV_NAME = "MiniGrid-DoorKey-5x5-v0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
loss_record = []
q_record = []
next_q_record = []

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()

        self.image = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.qnet = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        self.num_actions = num_outputs
    
    def forward(self, image, data):
        image = image.transpose(1, 3).transpose(2, 3)
        x1 = self.image(image)
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = data
        x = torch.cat((x1, x2), dim = 1)
        return self.qnet(x)
    
    def act(self, image, data, epsilon):
        if random.random() > epsilon:
            image = torch.FloatTensor(image).to(device).unsqueeze(0)
            data  = torch.FloatTensor(data).to(device).unsqueeze(0)
            action = self.forward(image, data).max(1)[1].detach().item()
            return action
        else:
            return np.random.randint(0, self.num_actions)

def update(model, target_model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return

    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    image_raw  = [np.expand_dims(d["image"],0) for d in batch.state]
    data_raw   = [np.expand_dims(d["data"],0) for d in batch.state]
    image_raw  = np.concatenate(image_raw)
    data_raw   = np.concatenate(data_raw)
    image      = torch.Tensor(image_raw)
    data       = torch.Tensor(data_raw)

    image_raw  = [np.expand_dims(d["image"],0) for d in batch.next_state]
    data_raw   = [np.expand_dims(d["data"],0) for d in batch.next_state]
    image_raw  = np.concatenate(image_raw)
    data_raw   = np.concatenate(data_raw)
    next_image = torch.Tensor(image_raw)
    next_data  = torch.Tensor(data_raw)

    action     = torch.Tensor(one_hot(np.array(batch.action), num_classes=7))
    reward     = torch.Tensor(batch.reward)
    done       = torch.Tensor(batch.done)

    q_value    = (model(image, data) * action).sum(dim=-1)
        
    next_q_value     = target_model(next_image, next_data).max(1)[0].detach()
    
    next_q_value = reward + GAMMA * next_q_value * (1 - done)
   
    loss_func = nn.SmoothL1Loss()
    loss = loss_func(q_value, next_q_value)
    # print(loss.item())
    loss_record.append(loss.item())
    # print(q_value.mean().item())
    q_record.append(q_value.mean().item())
    next_q_record.append(next_q_value.mean().item())
    
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def state_process(state, goal_pos, current_pos):
    direction = state['direction']
    s = np.append(direction, goal_pos)
    s = np.append(s, current_pos)
    image = state['image']
    state = {"image": image, "data": s}
    return state

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(0)
n = env.observation_space["image"].shape[0]
m = env.observation_space["image"].shape[1]
image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
act_shape = env.action_space.n

# GET GOAL POSE
for grid in env.grid.grid:
    if grid is not None and grid.type == "goal":
        goal_pose = grid.cur_pos

embedding_size = image_embedding_size + 2 + 2 + 1 # IMAGE SIZE + GOAL POSITION + AGENT POSITION + ORIENTATION
model_target = Net(embedding_size, act_shape).eval()
model        = Net(embedding_size, act_shape)
model_target.load_state_dict(model.state_dict())

if torch.cuda.is_available():
    model_target.to(device)
    model.to(device)

optimizer = optim.RMSprop(model.parameters(),lr=LR)

replay_buffer = ReplayMemory(MEMORY_CAPACITY)

epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 20000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

##############################################################
state = env.reset()  # MINIGRID OBSERVATION
state = state_process(state, goal_pose, env.agent_pos)
rewards = []
avg_rw = deque(maxlen=40)
episode_reward = 0
n_frame = 0
i_episode = 0
updated = False


for t in range(MAX_STEPS):
    a = model.act(state["image"], state["data"], epsilon_by_frame(n_frame))
    n_frame += 1
    next_state, reward, done, _ = env.step(a)
    next_state = state_process(next_state, goal_pose, env.agent_pos)

    episode_reward += reward
    replay_buffer.push(state, a, next_state, reward, done)
    state = next_state

    if done:
        state = env.reset()
        state = state_process(state, goal_pose, env.agent_pos)
        rewards.append(episode_reward)
        avg_rw.append(episode_reward)
        episode_reward = 0
        i_episode += 1
        updated = False
        model_target.load_state_dict(model.state_dict())

    if t % TRAIN_EVERY == 0:
        update(model, model_target, optimizer, replay_buffer, BATCH_SIZE)

    # if t % UPDATE_EVERY == 0 and not updated:
    if t % UPDATE_EVERY == 0:
        model_target.load_state_dict(model.state_dict())
        updated = True
        print(f'frames:{t}\teps:{i_episode}\tavg_rw:{np.mean(avg_rw)}\teps:{epsilon_by_frame(t)}')

plt.figure(1, figsize=(10,5))
rewards_smoothed = pd.Series(rewards).rolling(10, min_periods=10).mean()
plt.plot(rewards_smoothed)


plt.figure(2)
plt.plot(loss_record)

plt.figure(3)
plt.plot(q_record, 'b-*')
# plt.plot(next_q_record, 'r-h')

plt.show()
