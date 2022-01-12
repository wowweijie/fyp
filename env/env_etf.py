import os
import numpy as np
import pandas as pd

class EtfTradingEnv:
    def __init__(
        self,
        env_name,
        market_data_dim,
        lag,
        data_file,
        reward_scaling=2 ** -11,
        price_scaling=1e-3,
        min_order_val=10,
        max_order_val=1000,
        data_dir='./data/spdr500',
        start_capital=10**6
    ):
        self.env_name = env_name
        self.market_data_dim = market_data_dim
        self.lag = lag,
        self.data_file = data_file
        self.reward_scaling = reward_scaling
        self.price_scaling = price_scaling
        self.min_order_val = min_order_val
        self.max_order_val = max_order_val
        self.data_dir = data_dir
        self.start_capital = start_capital
        # state: position, bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume 
        self.state_dim = 1 + market_data_dim
        self.bid_close_idx = 3
        self.ask_close_idx = 7

        # reset
        self.position = 0
        self.open_position_val = 0
        self.step_idx = 0
        self.capital = self.start_capital

    def sample_tasks(self, num_tasks):
        return np.random.choice(np.array(os.listdir(self.data_dir)), size=num_tasks, replace=False)
    
    def load_data(self, filepath: str):
        header_list = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.read_csv(filepath, usecols=header_list)[header_list]
        
        datetime_col = pd.to_datetime(df['Gmt time'].str.slice(0,-7) , format='%d.%m.%Y %H:%M').dt
        df['hour'] = datetime_col.hour
        df['minute'] = datetime_col.hour
        df['dayWeek'] = datetime_col.dayofweek
        df.drop("Gmt time", axis=1, inplace=True)
        df.rename(columns={
            'Open' : 'open',
            'High' : 'high',
            'Low' : 'low',
            'Close' : 'close',
            'Volume' : 'vol'
        }, inplace=True)

    def reset_task(self, task):
        self.data_file = task
        self.data_df = self.load_data(self.data_dir + '/' + self.data_file)
        self.end_step = self.data_df.shape[0] - 1

    def step(self, actions):
        order_val = actions[0] * self.max_order_val
        curr_asset = self.open_position_val + self.capital

        if order_val < -self.min_order_val:
            bid_close = self.data_df.iloc[self.step_idx + self.lag, self.bid_close_idx]
            sell_qty = -order_val // bid_close
            if self.position > 0:
                self.capital += min(sell_qty, self.position) * bid_close
            self.position -= sell_qty

        elif order_val > self.min_order_val:
            ask_close = self.data_df.iloc[self.step_idx + self.lag, self.ask_close_idx]
            buy_qty = order_val // ask_close
            self.position += buy_qty
            self.capital -= buy_qty * ask_close
        
        

        self.step_idx += 1
        next_market_data = self.data_df.iloc[self.step_idx + self.lag].to_numpy()
        
        self.open_position_val = 0
        # calculate wealth
        # if there are open long positions
        if self.position > 0:
            # overwrite bid_close with new state 
            bid_close = next_market_data[self.bid_close_idx]
            self.open_position_val = self.position * bid_close

        elif self.position < 0:
            # overwrite ask_close with new state 
            ask_close = next_market_data[self.ask_close_idx]
            self.open_position_val = self.position * ask_close
        
        next_asset = self.open_position_val + self.capital
        reward = next_asset - curr_asset
        done = self.step_idx + self.lag == self.end_step

        if self.step_idx == 1 :
            self.market_data_state = self.lagged_market_data(self.step_idx + self.lag, self.lag)
        else:
            self.market_data_state = self.next_lagged_data(self.market_data_state, next_market_data)

        state = np.hstack(
            self.position,
            self.market_data_state,
        )

        return state, reward, done, dict()
    
    def reset(self):
        self.position = 0
        self.open_position_val = 0
        self.step_idx = 0
        self.capital = self.start_capital
    
    def lagged_market_data(self, curr_idx: int, lag: int):
        return self.data_df.iloc[curr_idx-lag: curr_idx+1].to_numpy().flatten()
    
    def next_lagged_data(self, curr_lagged_data: np.array, next_market_data: np.ndarray):
        lagged_len = len(curr_lagged_data)
        next_len = len(next_market_data)
        curr_lagged_data = np.roll(curr_lagged_data, -next_len)
        np.put(curr_lagged_data, range(lagged_len-next_len, lagged_len), next_market_data)
        return curr_lagged_data




        

    
        
        


        
            



