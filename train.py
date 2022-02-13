import gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from .env.env_etf import EtfTradingEnv
import os 
import argparse
import subprocess
import torch
import yaml
from logger import Logger
from datetime import datetime
import pytz

parser = argparse.ArgumentParser()
parser.add_argument('--remote', action="store_true", help='specify whether training is done on gcloud')
args = parser.parse_args()

with open(r'fyp/session.yaml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

sessionName = configs['sessionName']

timestamp = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%d_%m_%H-%M-%S")
logger = Logger(sessionName, timestamp, 'train')

logger.info(yaml.dump(configs))

cuda_availability = torch.cuda.is_available()
if cuda_availability:
    logger.info('CUDA enabled')
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
    logger.info('No CUDA')
    device = 'cpu'


if args.remote:
    subprocess.call([
        'gsutil', 'cp', '-r',
        # Storage path
        os.path.join('gs://', 'fyp-data-wj'),
        # Local path
        './data'
    ])

env = EtfTradingEnv(lag=configs['lag'])
env.reset_task('SPY.USUSD_Candlestick_1_M_01.01.2021-31.01.2021')
env = Monitor(env)
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=f'./logs/{sessionName}_{timestamp}/tb_logs/')
num_episode_train = 3
model.learn(total_timesteps=num_episode_train * env.get_episodic_step(), log_interval=400)
print(env.asset)
logger.info(f"env asset after training: {env.asset}")

trading_env = EtfTradingEnv(lag=configs['lag'])
trade_task = 'SPY.USUSD_Candlestick_1_M_01.02.2021-28.02.2021'
trading_env.reset_task(trade_task)
obs = trading_env.reset()
done = False 
logger.info("Start trading")
logger.info(f'trade task: {trade_task}')
while(not done):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = trading_env.step(action)
    trading_env.render()
    if done:
      obs = trading_env.reset()

logger.info(f"env asset after trading: {env.asset}")