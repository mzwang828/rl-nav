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

from common.multiprocessing_env import ParallelEnv
from algorithms.ppo import PPOAlgo

# Hyper Parameters
MINI_BATCH_SIZE  = 8
LR = 1e-3                   # learning rate
GAMMA = 0.99                # reward discount
TAU = 0.95                  # average parameter
PPO_EPOCHS = 4
MAX_FRAMES = int(1e5)
T_STEPS = 128               # steps per process before updating
L_STEPS = 10
MAX_GRAD_NORM = 0.5
ENV_NAME = "MiniGrid-Empty-5x5-v0"   #MiniGrid-Empty-5x5-v0   MiniGrid-DoorKey-5x5-v0  #MiniGrid-SimpleCrossingS9N1-v0
NUM_ENVS = 1
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
RECURRENCE = 1
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

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

class ACModel(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

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

    def forward(self, image, data, image_net):
        image = image.transpose(1, 3).transpose(2, 3)
        x1 = image_net(image)
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = data
        x = torch.cat((x1, x2), dim = 1)

        dist = self.actor(x)
        dist = Categorical(logits=F.log_softmax(dist, dim=1))

        value = self.critic(x).squeeze(1)
        return dist, value

def state_process(state, goal_pos, current_pos):
    direction = state['direction']
    s = np.append(direction, goal_pos)
    s = np.append(s, current_pos)
    image = state['image']
    return torch.Tensor(image).unsqueeze(0), torch.Tensor(s).unsqueeze(0)

def sub_goal_extract(n):
    x = n // 7
    y = n % 7
    return x - 3, y - 3

def compute_gae(values, rewards, masks):
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + DISCOUNT * values[step + 1] * masks[step] - values[step]
        gae = delta + DISCOUNT * GAE_LAMBDA * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages

def get_ppo_batch_index():
    global batch_num
    indexes = np.arange(0, T_STEPS, RECURRENCE)
    indexes = np.random.permutation(indexes)
    # Shift starting indexes by self.recurrence//2 half the time
    if batch_num % 2 == 1:
        indexes = indexes[(indexes + RECURRENCE) % T_STEPS != 0]
        indexes += RECURRENCE // 2
    batch_num += 1
    num_indexes = MINI_BATCH_SIZE // RECURRENCE
    batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

    return batches_starting_indexes

def ppo_update(model, optimizer, states_image, states_data, actions, values, log_probs, returns, advantages):
    for _ in range(PPO_EPOCHS):
        for indexs in get_ppo_batch_index():
            states_image_ = states_image[indexs]
            states_data_ = states_data[indexs]
            action = actions[indexs]
            old_log_probs = log_probs[indexs]
            return_ = returns[indexs]
            advantage = advantages[indexs]

            dist, value = model(states_image_, states_data_, image_net)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()

            value_clipped = return_ - advantage + torch.clamp(value - (return_ - advantage), -CLIP_EPS, CLIP_EPS)
            surr1 = (value - return_).pow(2)
            surr2 = (value_clipped - return_).pow(2)
            critic_loss = torch.max(surr1, surr2).mean()
            
            loss = VALUE_LOSS_COEF * critic_loss + actor_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(image_net.parameters()), MAX_GRAD_NORM)
            optimizer.step()

########################################################
if __name__  ==  "__main__":
    env  = gym.make(ENV_NAME)
    env.seed(0)
    n = env.observation_space["image"].shape[0]
    m = env.observation_space["image"].shape[1]
    image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
    embedding_size = image_embedding_size + 2 + 2 + 1
    act_shape = env.action_space.n

    image_net = ImageNet().to(device)
    hl_model = ACModel(embedding_size, 49).to(device)
    ll_model = ACModel(embedding_size, act_shape).to(device)
    hl_optimizer = torch.optim.Adam(list(hl_model.parameters())+list(image_net.parameters()), lr = 0.001, eps=1e-8)
    ll_optimizer = torch.optim.Adam(list(ll_model.parameters())+list(image_net.parameters()), lr = 0.001, eps=1e-8)
    #####################################################
    state = env.reset()
    avg_rw = deque(maxlen= 40)
    n_frame = 0
    i_episode = 0
    episode_reward = 0
    done = False
    for grid in env.grid.grid:
        if grid is not None and grid.type == "goal":
            goal_pose = grid.cur_pos

    for t in range(MAX_FRAMES):
        # variables to record experience
        log_probs_h = []
        values_h = []
        states_image_h = []
        states_data_h = []
        actions_h = []
        rewards_h = []
        masks_h = []
        log_probs_l = []
        values_l = []
        states_image_l = []
        states_data_l = []
        actions_l = []
        rewards_l = []
        masks_l = []
        ll_update_counter = 0

        for i in range(T_STEPS):
            ll_t = 0
            extrinsic_reward = 0

            image_processed, data_processed = state_process(state, goal_pose, env.agent_pos)
            with torch.no_grad():
                dist, value = hl_model(image_processed, data_processed, image_net)

            action = dist.sample()
            sub_x, sub_y = sub_goal_extract(action.cpu().numpy())
            subgoal = np.array([sub_x, sub_y])
            subgoal_abs = env.agent_pos + np.array([sub_x, sub_y])

            while not done and np.any(env.agent_pos != subgoal_abs) and ll_t < L_STEPS:
                image_processed_l, data_processed_l = state_process(state, subgoal, env.agent_pos)
                with torch.no_grad():
                    dist_l, value_l = ll_model(image_processed_l, data_processed_l, image_net)
                action_l = dist_l.sample()
                next_state, reward, done, _ = env.step(action_l.cpu().numpy())

                extrinsic_reward += reward
                episode_reward += reward
                intrinsic_reward = 1.0 if np.all(env.agent_pos == subgoal_abs) else 0.0
                intrinsic_done = True if np.all(env.agent_pos == subgoal_abs) else False

                log_prob_l = dist_l.log_prob(action_l)
                log_probs_l.append(log_prob_l)
                values_l.append(value_l)
                rewards_l.append(torch.Tensor([intrinsic_reward]))
                masks_l.append(torch.Tensor(1-np.stack([done])))
                states_image_l.append(image_processed_l)
                states_data_l.append(data_processed_l)
                actions_l.append(action_l)
                state = next_state
                ll_t += 1
                ll_update_counter += 1

                #update lower network every T_STEPS
                if ll_update_counter % T_STEPS == 0:
                    with torch.no_grad():
                        next_image_processed, next_data_processed = state_process(next_state, subgoal, env.agent_pos)
                        _, next_value = ll_model(next_image_processed, next_data_processed, image_net)
                    values_l = values_l + [next_value]
                    returns, advantages = compute_gae(values_l, rewards_l, masks_l)

                    returns   = torch.cat(returns).detach()
                    log_probs = torch.cat(log_probs_l).detach()
                    values    = torch.cat(values_l).detach()
                    states_image    = torch.cat(states_image_l)
                    states_data    = torch.cat(states_data_l)
                    actions   = torch.cat(actions_l)
                    advantages = torch.cat(advantages)

                    ppo_update(ll_model, ll_optimizer, states_image, states_data, actions, values, log_probs, returns, advantages)

                    log_probs_l = []
                    values_l = []
                    states_image_l = []
                    states_data_l = []
                    actions_l = []
                    rewards_l = []
                    masks_l = []

            log_prob = dist.log_prob(action)
            log_probs_h.append(log_prob)
            values_h.append(value)
            rewards_h.append(torch.Tensor([extrinsic_reward]))
            masks_h.append(1 - torch.Tensor([done]))
            states_image_h.append(image_processed)
            states_data_h.append(data_processed)
            actions_h.append(action)
            state = next_state

            if done:
                state = env.reset()
                done = False
                avg_rw.append(episode_reward)
                episode_reward = 0
                i_episode += 1

        with torch.no_grad():
            next_image_processed, next_data_processed = state_process(next_state, goal_pose, env.agent_pos)
            _, next_value = hl_model(next_image_processed, next_data_processed, image_net)
        values_h = values_h + [next_value]
        returns, advantages = compute_gae(values_h, rewards_h, masks_h)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs_h).detach()
        values    = torch.cat(values_h).detach()
        states_image    = torch.cat(states_image_h)
        states_data    = torch.cat(states_data_h)
        actions   = torch.cat(actions_h)
        advantages = torch.cat(advantages)

        ppo_update(hl_model, hl_optimizer, states_image, states_data, actions, values, log_probs, returns, advantages)

        print(f'episode:{i_episode}\tavg_rw:{np.mean(avg_rw)}')