import math
import random

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

from common.multiprocessing_env import ParallelEnv
from algorithms.ppo import PPOAlgo

# Hyper Parameters
MINI_BATCH_SIZE  = 256
LR = 1e-3                   # learning rate
GAMMA = 0.99                # reward discount
TAU = 0.95                  # average parameter
PPO_EPOCHS = 4
MAX_FRAMES = 1e5
T_STEPS = 128               # steps per process before updating
MAX_GRAD_NORM = 0.5
ENV_NAME = "MiniGrid-Empty-8x8-v0"   #MiniGrid-Empty-5x5-v0   MiniGrid-DoorKey-5x5-v0  #MiniGrid-SimpleCrossingS9N1-v0
NUM_ENVS = 16

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

        value = self.critic(x).squeeze(1)
        return dist, value
    
def state_process(state, goal_pos, current_pos):
    images = []
    datas = []
    for i in range(len(state)) :
        direction = state[i]['direction']
        s = np.append(direction, goal_pos[i])
        s = np.append(s, current_pos[i])
        image = state[i]['image']
        images.append(image)
        datas.append(s)
    return torch.Tensor(images), torch.Tensor(datas)


def test_env(acmodel, test_env, vis=False):
    state = test_env.reset()
    if vis: test_env.render()
    for grid in test_env.grid.grid:
        if grid is not None and grid.type == "goal":
            goal = grid.cur_pos
    done = False
    total_reward = 0
    while not done:
        image = torch.Tensor(state['image']).unsqueeze(0)
        direction = state['direction']
        data = np.append(direction, goal)
        data = np.append(data, test_env.agent_pos)
        data = torch.Tensor(data).unsqueeze(0)
        dist, _ = acmodel(image, data)
        next_state, reward, done, _ = test_env.step(dist.sample().cpu().numpy())
        state = next_state
        if vis: test_env.render()
        total_reward += reward
    return total_reward

def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    return env
###########################################

if __name__  ==  "__main__":
    envs = [make_env(ENV_NAME, 1 + 10000*i) for i in range(NUM_ENVS)]
    print("Environments Loaded!\n")
    ########################################
    env  = gym.make(ENV_NAME)
    env.seed(0)
    n = env.observation_space["image"].shape[0]
    m = env.observation_space["image"].shape[1]
    image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
    embedding_size = image_embedding_size + 2 + 2 + 1
    act_shape = env.action_space.n
    model = ActorCritic(embedding_size, act_shape).to(device)
    ###########################################
    ppo = PPOAlgo(envs, model, None, device, T_STEPS, state_process)
    frame_idx = 0

    while frame_idx < MAX_FRAMES:

        states_image, states_data, actions, values, log_probs, returns, advantages = ppo.collect_experience()
        ppo.update_params(states_image, states_data, actions, values, log_probs, returns, advantages)
        frame_idx += T_STEPS * NUM_ENVS

        test_reward = np.mean([test_env(model, env, False) for _ in range(10)])
        print(f'frames:{frame_idx}\tavg_rw:{test_reward}')
