import gym
from stable_baselines3 import A2C
from env.env_etf import EtfTradingEnv
from logger import Logger
import logging
import os 

env = EtfTradingEnv(2)
env.reset_task('')
model = A2C('MlpPolicy', env, verbose=1)
num_episode_train = 2
model.learn(total_timesteps=10000)