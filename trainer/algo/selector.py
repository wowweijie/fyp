from stable_baselines3 import A2C, PPO
from trainer.sb3_contrib.trpo.trpo import TRPO
# from trainer.maml.metalearners.maml_trpo import MAMLTRPO
# from trainer.logger import Logger

class RlAlgoSelector():
    @classmethod
    def init(cls, algo_name: str, **kwargs):
        if algo_name == 'A2C':
            return A2C(**kwargs)
        elif algo_name == 'PPO':
            return PPO(**kwargs)
        elif algo_name == 'TRPO':
            return TRPO(**kwargs)
        # elif algo_name == 'MAMLTRPO':
        #     return MAMLTRPO(env=kwargs['env'], logger = Logger())
