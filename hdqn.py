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

from IPython.display import clear_output
import matplotlib.pyplot as plt

from misc_utils import one_hot, Transition, ReplayMemory

# Hyper Parameters
BATCH_SIZE = 64
LR = 1e-5                    # learning rate
GAMMA = 0.99                 # reward discount
MEMORY_CAPACITY = 10000
MAX_EPISODES = 3000
MAX_STEPS = int(1e6)
T_STEPS = 8
UPDATE_EVERY = int(1e3)
TRAIN_EVERY = 4
ENV_NAME = "MiniGrid-SimpleCrossingS9N1-v0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)
    
class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        self.num_actions = num_outputs
    
    def forward(self, image, data, cnn_net):
        image = image.transpose(1, 3).transpose(2, 3)
        x1 = cnn_net(image)
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = data
        x = torch.cat((x1, x2), dim = 1)
        return self.layers(x)
    
    def act(self, image, data, cnn_net, epsilon):
        if random.random() > epsilon:
            image = torch.Tensor(image).unsqueeze(0)
            data  = torch.Tensor(data).unsqueeze(0)
            action = self.forward(image, data, cnn_net).max(1)[1].detach().item()
            return action
        else:
            return random.randrange(self.num_actions)

def update(model, target_model, cnn_model, optimizer, replay_buffer, batch_size, class_size):
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

    action     = torch.Tensor(one_hot(np.array(batch.action), num_classes=class_size))
    reward     = torch.Tensor(batch.reward)
    done       = torch.Tensor(batch.done)

    q_value    = (model(image, data, cnn_model) * action).sum(dim=-1)
    next_q_value     = target_model(next_image, next_data, cnn_model).max(1)[0].detach()
    next_q_value = reward + GAMMA * next_q_value * (1 - done)
   
    loss_func = nn.MSELoss()
    loss = loss_func(q_value, next_q_value)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def state_process(state, goal_pos, current_pos):
    direction = state['direction']
    s = np.append(direction, goal_pos)
    s = np.append(s, current_pos)
    image = state['image']
    state = {"image": image, "data": s}
    return state

################################################
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

sub_goal_dict = {'0':(-2,-2),
                 '1':(-2,-1),
                 '2':(-2,-0),
                 '3':(-2,1),
                 '4':(-2,2),
                 '5':(-1,-2),
                 '6':(-1,-1),
                 '7':(-1,0),
                 '8':(-1,1),
                 '9':(-1,2),
                 '10':(0,-2),
                 '11':(0,-1),
                 '12':(0,0),
                 '13':(0,1),
                 '14':(0,2),
                 '15':(1,-2),
                 '16':(1,-1),
                 '17':(1,0),
                 '18':(1,1),
                 '19':(1,2),
                 '20':(2,-2),
                 '21':(2,-1),
                 '22':(2,0),
                 '23':(2,1),
                 '24':(2,2)}

embedding_size = image_embedding_size + 2 + 2 + 1 # IMAGE SIZE + GOAL POSITION + AGENT POSITION + ORIENTATION

hl_model = QNet(embedding_size, 25)
hl_model_target = QNet(embedding_size, 25).eval()
hl_model_target.load_state_dict(hl_model.state_dict())
ll_model = QNet(embedding_size, act_shape)
ll_model_target = QNet(embedding_size, act_shape).eval()
ll_model_target.load_state_dict(ll_model.state_dict())
image_net = ImageNet()

if torch.cuda.is_available():
    image_net.to(device)
    hl_model.to(device)
    hl_model_target.to(device)
    ll_model.to(device)
    ll_model_target.to(device)

hl_optimizer = optim.RMSprop(list(hl_model.parameters())+list(image_net.parameters()),lr=LR)
ll_optimizer = optim.RMSprop(list(ll_model.parameters())+list(image_net.parameters()),lr=LR)

hl_replay_buffer = ReplayMemory(MEMORY_CAPACITY)
ll_replay_buffer = ReplayMemory(MEMORY_CAPACITY)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 15000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

##########################################################################3
state_env = env.reset()  # MINIGRID OBSERVATION
state = state_process(state_env, goal_pose, env.agent_pos)
rewards = []
avg_rw = deque(maxlen=40)
episode_reward = 0
n_frame = 0
i_episode = 0
done = False

for t in range(MAX_STEPS):
    sub_goal_i = hl_model.act(state["image"], state["data"], image_net, epsilon_by_frame(t))
    sub_goal = sub_goal_dict[str(sub_goal_i)]
    abs_sub_goal = tuple(map(sum, zip(env.agent_pos, sub_goal)))
    ll_t = 0
    extrinsic_reward = 0

    sub_state = state_process(state_env, abs_sub_goal, env.agent_pos)

    while not done and np.any(env.agent_pos != abs_sub_goal) and ll_t < T_STEPS:
        action = ll_model.act(sub_state["image"], sub_state["data"], image_net, epsilon_by_frame(t))
        next_state_env, reward, done, _ = env.step(action)
        next_sub_state = state_process(next_state_env, abs_sub_goal, env.agent_pos)

        episode_reward += reward
        extrinsic_reward += reward
        intrinsic_reward = 1.0 if np.all(env.agent_pos == abs_sub_goal) else 0.0

        ll_replay_buffer.push(sub_state, action, next_sub_state, intrinsic_reward, done)
        sub_state = next_sub_state

        update(ll_model, ll_model_target, image_net, ll_optimizer, ll_replay_buffer, BATCH_SIZE, 7)
        update(hl_model, hl_model_target, image_net, hl_optimizer, hl_replay_buffer, BATCH_SIZE, 25)
        ll_t += 1
        state_env = next_state_env

    next_state = state_process(state_env, goal_pose, env.agent_pos)
    hl_replay_buffer.push(state, sub_goal_i, next_state, extrinsic_reward, done)
    state = next_state

    if done:
        state_env = env.reset()
        state = state_process(state_env, goal_pose, env.agent_pos)
        done = False
        rewards.append(episode_reward)
        episode_reward = 0
        i_episode += 1
        print(i_episode)

plt.figure(figsize=(10,5))
rewards_smoothed = pd.Series(rewards).rolling(10, min_periods=10).mean()
plt.plot(rewards_smoothed)
plt.show()