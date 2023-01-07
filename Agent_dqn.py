
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from random import random, sample
from collections import deque
from utils import FrameStackingAndResizingEnv
from dqn_model import DQN
from torch.optim import Adam
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.idx = 0

    def insert(self, sars):
        self.buffer.append(sars)

    def sample(self, num_samples):
        if num_samples > len(self.buffer):
            return sample(self.buffer, len(self.buffer))
        return sample(self.buffer, num_samples)

learning_rate = 0.0001


class memory:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())


def train_step(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  
    
    model.optimizer.zero_grad()
    qvals = model(cur_states) 
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    # Calculate the predicted Q values
    qvals_pred = torch.sum(qvals * one_hot_actions, -1)

    # Calculate the target Q values
    qvals_target = rewards.squeeze() + mask[:, 0] * qvals_next * gamma

    # Calculate the loss using mean squared error
    loss = ((qvals_pred - qvals_target) ** 2).mean()

    # Backpropagate the loss and update the model weights
    loss.backward()
    model.optimizer.step()
    return loss



def train(file, name='breakout',  device="cuda",test=False):
    
    min_rb_size = 50000
    sample_size = 32
    lr = 0.0001
    num_iterations = 15_000_000
    eps_decay = 0.999999

    env = gym.make("Breakout-v0",)
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)
    last_observation = env.reset()
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    if test:
        model.load_state_dict(torch.load(file))
    target = DQN(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(model, target)

    rb = ReplayBuffer()
    steps_since_train = 0
    loss_set = []

    step_num = -1 * min_rb_size
    episode_rewards = []
    rolling_reward = 0
    mean_reward_per_episode = 0
    for i in range(num_iterations):
        observation = env.reset()
        done = False
        rolling_reward = 0
        while not done:

            eps = eps_decay ** (step_num)
           
            if random() < eps:
                action = (env.action_space.sample()) 
            else:
                action = model(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item() 
            
            observation, reward, done, info = env.step(action)
            rolling_reward += reward

            rb.insert(memory(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                # print("average_reward : ",episode_rewards)
                
                observation = env.reset()
                loss = train_step(model, rb.sample(sample_size), target, env.action_space.n, device).detach().cpu().numpy()
                loss_set.append(loss.tolist())

        print("number of iters :", i)
        print("rewards (unclipped): ",rolling_reward)
       
    env.close()


    fig, axs = plt.subplots(2)
    fig.suptitle('loss vs number of episodes and rewards per episodes plot')
    axs[0].plot(range(1,len(loss_set)+1),loss_set)
    axs[1].plot(range(1,len(episode_rewards)+1),episode_rewards)
    plt.show()



def test(file, name='breakout', device="cuda", test=True):
    num_iterations = 100
    initiate_sequence = 5

    env = gym.make("Breakout-v0", render_mode="human")
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)

    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    if file:
        model.load_state_dict(torch.load(file))
    target = DQN(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(model, target)

    rb = ReplayBuffer()
    episode_rewards = []

    for _ in range(num_iterations):
        observation = env.reset()
        last_observation = observation
        done = False
        rolling_reward = 0
        while not done:
            env.render(mode='rgb_array')
            eps = 0

            if random() < eps or (test and initiate_sequence > 0):
                action = env.action_space.sample()
                initiate_sequence -= 1
            else:
                action = model(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item() 

            observation, reward, done, info = env.step(action)
            rolling_reward += reward
            rb.insert(memory(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                print("reward per episode, unclipped :", rolling_reward)
    env.close()
    print("Mean reward for 100 episodes: ", np.mean(episode_rewards))



if __name__ == "__main__":
  
    # train(file=None)

    test(file = 'trained_DQN.pth')

