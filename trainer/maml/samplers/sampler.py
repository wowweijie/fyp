import gym


class make_env(object):
    def __init__(self, env_func, env_kwargs):
        self.env_func = env_func
        self.env_kwargs = env_kwargs

    def __call__(self):
        return self.env_func(**self.env_kwargs)

class Sampler(object):
    def __init__(self,
                 env,
                 batch_size,
                 policy,
                 seed=None):
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)
