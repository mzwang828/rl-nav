import numpy as np
import torch
from common.multiprocessing_env import ParallelEnv

class PPOAlgo:
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, preprocess_obss=None,
                 discount=0.99, lr=0.001, gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, 
                 max_grad_norm=0.5, recurrence=4, adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, 
                 reshape_reward=None):

        # store parameters
        self.envs = ParallelEnv(envs)
        self.num_procs = len(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss
        self.reshape_reward = reshape_reward
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_num = 0

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

        self.state = self.envs.reset()
        self.goal = self.envs.get_goal()

    def collect_experience(self):
        # variables to record experience
        log_probs = []
        values = []
        states_image = []
        states_data = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        # perform environment interaction multi-program and store experience
        for i in range(self.num_frames_per_proc):
            image_processed, data_processed = self.preprocess_obss(self.state, self.goal, self.envs.agent_pose())
            with torch.no_grad():
                dist, value = self.acmodel(image_processed, data_processed)
            
            action = dist.sample()
            next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
            
            self.goal = self.envs.get_goal()
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.Tensor(reward))
            masks.append(torch.Tensor(1-np.stack(done)))
            states_image.append(image_processed)
            states_data.append(data_processed)
            actions.append(action)
            self.state = next_state
        
        # calculate returns and advantages for experience
        with torch.no_grad():
            next_image_processed, next_data_processed = self.preprocess_obss(next_state, self.goal, self.envs.agent_pose())
            _, next_value = self.acmodel(next_image_processed, next_data_processed)

        values = values + [next_value]

        returns, advantages = self.compute_gae(values, rewards, masks)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states_image    = torch.cat(states_image)
        states_data    = torch.cat(states_data)
        actions   = torch.cat(actions)
        advantages = torch.cat(advantages)
        
        return states_image, states_data, actions, values, log_probs, returns, advantages

    def update_params(self, states_image, states_data, actions, values, log_probs, returns, advantages):
        for _ in range(self.epochs):
            for indexs in self.get_ppo_batch_index():
                states_image_ = states_image[indexs]
                states_data_ = states_data[indexs]
                action = actions[indexs]
                old_log_probs = log_probs[indexs]
                return_ = returns[indexs]
                advantage = advantages[indexs]

                dist, value = self.acmodel(states_image_, states_data_)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                actor_loss  = - torch.min(surr1, surr2).mean()

                value_clipped = return_ - advantage + torch.clamp(value - (return_ - advantage), -self.clip_eps, self.clip_eps)
                surr1 = (value - return_).pow(2)
                surr2 = (value_clipped - return_).pow(2)
                critic_loss = torch.max(surr1, surr2).mean()
                
                loss = self.value_loss_coef * critic_loss + actor_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def get_ppo_batch_index(self):
        indexes = np.arange(0, self.num_frames_per_proc * self.num_procs, self.recurrence)
        indexes = np.random.permutation(indexes)
        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def compute_gae(self, values, rewards, masks):
        gae = 0
        returns = []
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount * values[step + 1] * masks[step] - values[step]
            gae = delta + self.discount * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
        return returns, advantages