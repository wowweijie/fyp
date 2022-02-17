from distutils.command.config import config
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from trainer.env.env_etf import EtfTradingEnv
import os 
import argparse
import subprocess
import torch
import yaml
from trainer.logger import Logger
from datetime import datetime
import pytz

parser = argparse.ArgumentParser()
parser.add_argument('--remote', action="store_true", help='specify whether training is done on gcloud')
parser.add_argument('--job-dir', action="store", type=str, help='specify gcloud storage job dir path')
args = parser.parse_args()

globals()['DATA_PATH'] = '/tmp' if args.remote else './' 
globals()['CONFIG_PATH'] = '/tmp' if args.remote else '../' 
DATA_PATH = globals()['DATA_PATH']
CONFIG_PATH = globals()['CONFIG_PATH']
 
print(f"Current Directory: {os.getcwd()}")
print(f"ls: {[f for f in os.listdir('.')]}")
if args.remote:
    os.mkdir(os.path.join(DATA_PATH, 'data'))
    subprocess.call([
        'gsutil', 'cp', '-r',
        # Storage path
        'gs://fyp-data-wj/*',
        # Local path
        os.path.join(DATA_PATH, 'data')
    ])
    print(f"args.job_dir: {args.job_dir}")  
    os.mkdir(os.path.join(DATA_PATH, 'config'))
    subprocess.call([
        'gsutil', 'cp',
        # Storage path
        os.path.join(args.job_dir, "config", "*.yaml"),
        # Local path
        os.path.join(DATA_PATH, 'config')
    ])
    with open(os.path.join(DATA_PATH, 'config', 'session.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)                                                                                                                                                               
else:
    with open(r'../session.yaml') as file:
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

print(f"/tmp: {[f for f in os.listdir('/tmp')]}")
print(f"/tmp/data: {[f for f in os.listdir('/tmp/data')]}")
print(f"/tmp/data/spdr500: {[f for f in os.listdir('/tmp/data/spdr500')]}")
env = EtfTradingEnv(lag=configs['lag'], data_dir=os.path.join(DATA_PATH, 'data/spdr500'))
env.reset_task('SPY.USUSD_Candlestick_1_M_01.01.2021-31.01.2021')
env = Monitor(env)
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=f'./logs/{sessionName}_{timestamp}/tb_logs/')
num_episode_train = 3
model.learn(total_timesteps=num_episode_train * env.get_episodic_step(), log_interval=400)
print(env.asset)
logger.info(f"env asset after training: {env.asset}")

trading_env = EtfTradingEnv(lag=configs['lag'], data_dir=os.path.join(DATA_PATH, 'data/spdr500'))
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