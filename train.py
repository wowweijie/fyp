import gym
from stable_baselines3 import A2C
from env.env_etf import EtfTradingEnv
import os 
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--remote', action="store_true", help='specify whether training is done on gcloud')
parser.add_argument('--lag', type=int, help='specify market data lag on env. 0 means no lagged data, only current data')
args = parser.parse_args()

if args.remote:
    subprocess.call([
        'gsutil', 'cp', '-r',
        # Storage path
        os.path.join('gs://', 'fyp-data-wj'),
        # Local path
        './data'
    ])

env = EtfTradingEnv(lag=args.lag)
env.reset_task('SPY.USUSD_Candlestick_1_M_01.01.2021-31.01.2021')
model = A2C('MlpPolicy', env, verbose=1)
num_episode_train = 100
model.learn(total_timesteps=3 * env.get_episodic_step())
print(env.asset)

trading_env = EtfTradingEnv(lag=args.lag)
trading_env.reset_task('SPY.USUSD_Candlestick_1_M_01.02.2021-28.02.2021')
obs = trading_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = trading_env.step(action)
    trading_env.render()
    if done:
      obs = trading_env.reset()