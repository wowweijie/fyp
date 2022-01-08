import os
import numpy as np

class EtfTradingEnv:
    def __init__(
        self,
        env_name,
        market_data_dim,
        feature_generators,
        reward_scaling,
        price_scaling,
        min_order_qty,
        max_order_qty,
        data_file,
        data_dir='./data/index/spdr500',
        start_capital=10**6
    ):
        self.env_name = env_name
        self.market_data_dim = market_data_dim
        self.feature_generators = feature_generators
        self.reward_scaling = reward_scaling
        self.price_scaling = price_scaling
        self.min_order_qty = min_order_qty
        self.max_order_qty = max_order_qty
        self.data_df = data_df
        self.data_dir = data_dir
        self.start_capital = start_capital
        # state: position, bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume 
        self.state_dim = 1 + market_data_dim 

    def sample_tasks(self, num_tasks):
        return np.random.choice(np.array(os.listdir(self.data_dir), size=num_tasks))
    
    def step(self, actions):
        actions = (actions * self.max_order_qty).astype(int)
        
        


        
            



