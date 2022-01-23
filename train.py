import gym
from stable_baselines3 import A2C
from env.env_etf import EtfTradingEnv
from logger import Logger
import logging
import os 

env = EtfTradingEnv(2)
env.reset_task('SPY.USUSD_Candlestick_1_M_01.01.2021-31.01.2021')
model = A2C('MlpPolicy', env, verbose=1)
num_episode_train = 3
model.learn(total_timesteps=3 * env.get_episodic_step())
print(env.asset)