from stable_baselines3.common.monitor import Monitor
from trainer.env.env_etf import EtfTradingEnv
from trainer.algo.selector import RlAlgoSelector
from trainer.configs import Config
import os 
import argparse
import subprocess
import torch
import yaml
import sys
from trainer.logger import Logger
from trainer.utils import action_mask
from datetime import datetime
import pytz
from timeit import default_timer as timer
import multiprocessing as mp

print(f"system info: {sys.version}")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', action="store_true", help='specify whether training is done on gcloud')
    parser.add_argument('--config-ver', action="store", type=str, default = None, help='specify which session yaml config to use')
    parser.add_argument('--job-dir', action="store", type=str, help='specify gcloud storage job dir path')
    args = parser.parse_args()

    globals()['DATA_PATH'] = '/tmp' if args.remote else './' 
    DATA_PATH = globals()['DATA_PATH']

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
        with open(os.path.join(DATA_PATH, 'config', f'session_remote{args.config_ver}.yaml')) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)                                                                                                                                                               
    else:
        with open(f'../session_local{args.config_ver}.yaml') as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

    globals()['configs'] = configs
    Config.set_configs(configs)

    sessionName = configs['sessionName']

    timestamp = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%d_%m_%H-%M-%S")
    logger = Logger(sessionName, timestamp, 'train')

    logger.info(yaml.dump(configs))

    cuda_availability = torch.cuda.is_available()
    if cuda_availability and not configs['force-cpu']:
        logger.info('CUDA enabled')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        logger.info('No CUDA')
        device = 'cpu'

    Config.configs['device'] = device
    if configs['model'].get('maml'):
        Config.configs['model']['maml']['device'] = device
 
    env_configs = {'lag': configs['lag'], 'data_dir': os.path.join(DATA_PATH, f"data/{configs['sessionName']}"), 'task_distribution': configs['train_tasks']}
    env = EtfTradingEnv(lag=configs['lag'], data_dir=os.path.join(DATA_PATH, f"data/{configs['sessionName']}"))
    train_tasks = configs['train_tasks']
    if configs['model'].get('maml') is None and train_tasks:
        env.reset_task(*train_tasks)
    elif configs['model'].get('maml') and train_tasks:
        env.set_task_distribution(*train_tasks)
        logger.info('Task distribution set up')
    else:
        logger.error('train_tasks must be in config')
        sys.exit()

    # env = Monitor(env)
    # model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=f'./logs/{sessionName}_{timestamp}/tb_logs/')
    algo = configs['model']['algo']
    model = RlAlgoSelector.init(algo, policy='MlpPolicy', env=env, env_kwargs=env_configs, 
     learning_rate=configs['model']['fast-lr'], gamma=configs['model']['gamma'],
     cg_max_steps=configs['model']['cg-iters'], cg_damping=configs['model']['cg-damping'],
     line_search_shrinking_factor=configs['model']['ls-backtrack-ratio'], line_search_max_iter=configs['model']['ls-max-steps'],
     gae_lambda=configs['model']['gae-lambda'], target_kl=configs['model']['max-kl'],
     verbose=1, tensorboard_log=f'./logs/{sessionName}_{timestamp}/tb_logs/')

    num_episode_train = configs['num_episode_train']
    start_learn = timer()
    model.learn(total_timesteps=num_episode_train * env.get_episodic_step(), log_interval=400)
    end_learn = timer()
    logger.info(f"train time: {end_learn - start_learn}")
    logger.save_model(model)
    logger.info("Evaluate training")
    obs = env.reset()
    done = False
    while(not done):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

    logger.info(f"env asset after training without mask: {env.asset}")
    logger.csv(env.performance, "train_perf_no_mask")

    obs = env.reset()
    done = False
    while(not done):
        action, _ = model.predict(obs, deterministic=True)
        action_mask(action, configs['action_mask']['upper_thres'], configs['action_mask']['lower_thres'])    
        obs, reward, done, info = env.step(action)
        env.render()

    logger.info(f"env asset after training with mask: {env.asset}")
    logger.csv(env.performance, "train_perf_with_mask")

    trading_env = EtfTradingEnv(lag=configs['lag'], data_dir=os.path.join(DATA_PATH, f"data/{configs['sessionName']}"))
    trade_tasks = configs['trade_tasks']
    trading_env.reset_task(*trade_tasks)
    logger.info("Start trading")
    logger.info(f'trade task: {trade_tasks}')
    obs = trading_env.reset()
    done = False 
    while(not done):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = trading_env.step(action)
        trading_env.render()

    logger.info(f"env asset after trading without mask: {trading_env.asset}")
    logger.csv(trading_env.performance, "trade_perf_no_mask")

    obs = trading_env.reset()
    done = False 
    while(not done):
        action, _ = model.predict(obs, deterministic=True)
        action_mask(action, configs['action_mask']['upper_thres'], configs['action_mask']['lower_thres'])
        obs, reward, done, info = trading_env.step(action)
        trading_env.render()

    logger.info(f"env asset after trading with mask: {trading_env.asset}")
    logger.csv(trading_env.performance, "trade_perf_with_mask")

    if args.remote:
        subprocess.call([
            'gsutil', '-m', 'cp', '-r',
            # logs
            f'./logs/{sessionName}_{timestamp}/*',
            # Jobs storage dir
            f'{args.job_dir}/logs'
        ])