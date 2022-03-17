from stable_baselines3 import A2C, PPO
from trainer.sb3_contrib.trpo.trpo import TRPO
from trainer.maml.metalearners.maml_trpo import MAMLTRPO
from trainer.logger import Logger
from trainer.configs import Config

class RlAlgoSelector():
    @classmethod
    def init(cls, algo_name: str, **kwargs):
        if algo_name == 'A2C':
            kwargs.pop('env_kwargs')
            return A2C(**kwargs)
        elif algo_name == 'PPO':
            kwargs.pop('env_kwargs')
            return PPO(**kwargs)
        elif algo_name == 'TRPO':
            kwargs.pop('env_kwargs')
            return TRPO(**kwargs)
        elif algo_name == 'MAMLTRPO':
            return MAMLTRPO(env=kwargs['env'], env_kwargs=kwargs['env_kwargs'], logger = Logger(), device=Config.configs['device'])
