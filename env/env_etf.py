import gym
import os
import numpy as np
import pandas as pd
from logger import Logger 


class EtfTradingEnv(gym.Env):
    def __init__(
        self,
        lag,
        env_name="etf_spdr500",
        reward_scaling=2 ** -11,
        price_scaling=1e-3,
        min_order_val=10,
        max_order_val=1000,
        data_dir='./data/spdr500',
        start_capital=10**6
    ):
        self.env_name = env_name,
        self.lag = lag,
        self.reward_scaling = reward_scaling
        self.price_scaling = price_scaling
        self.min_order_val = min_order_val
        self.max_order_val = max_order_val
        self.data_dir = data_dir
        self.start_capital = start_capital
        # state: position, bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume 
        self.bid_close_idx = 3
        self.ask_close_idx = 7

        self.logger = Logger("etf_spdr500")

        # reset
        self.position = 0
        self.open_position_val = 0
        self.step_idx = 0
        self.capital = self.start_capital
        self.asset = self.start_capital

    def sample_tasks(self, num_tasks):
        return np.random.choice(np.array(os.listdir(self.data_dir)), size=num_tasks, replace=False)
    
    def load_data(self, filepath: str, askOrBid: str):
        header_list = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.read_csv(filepath, usecols=header_list)[header_list]
        
        datetime_col = pd.to_datetime(df['Gmt time'].str.slice(0,-7) , format='%d.%m.%Y %H:%M')
        datetime_prop = datetime_col.dt
        df['hour'] = datetime_prop.hour
        df['minute'] = datetime_prop.minute
        df['dayWeek'] = datetime_prop.dayofweek
        df['timestamp'] = datetime_col
        df.drop("Gmt time", axis=1, inplace=True)
        df.set_index('timestamp', drop=True, inplace=True)
        df.rename(columns={
            'Open' : askOrBid + '_open',
            'High' : askOrBid + '_high',
            'Low' : askOrBid + '_low',
            'Close' : askOrBid + '_close',
            'Volume' : askOrBid + '_vol'
        }, inplace=True)

        return df

    def load_task(self, source):
        df1 = None
        df2 = None
        for file in os.listdir(source):
            if "BID" in file:
                if df1 is None:
                    df1 = self.load_data(source + '/' + file, 'bid')
                    print("loaded " + file)
                else:
                    print("Unable to overwrite existing bid data")
            elif "ASK" in file:
                if df2 is None:
                    df2 = self.load_data(source + '/' + file, 'ask')
                    print("loaded " + file)
                else:
                    print("Unable to overwrite existing ask data")
            else:
                print("skipping " + file)
        return pd.concat([df1, df2], axis=1)

    def reset_task(self, task):
        self.data_df = self.load_task(self.data_dir + '/' + task)
        self.end_step = self.data_df.shape[0] - 1
        self.market_data_dim = self.data_df.shape[1]

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
        
        # self.asset is the next state asset
        self.asset = self.open_position_val + self.capital
        reward = self.asset - curr_asset
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
        self.asset = self.start_capital
    
    def render(self):
        if self.step_idx == 0:
            self.logger.info(self.env_name + " :: Start session")
        else:
            self.logger.info(f"{self.step_idx},{self.asset}")
    
    def lagged_market_data(self, curr_idx: int, lag: int):
        return self.data_df.iloc[curr_idx-lag: curr_idx+1].to_numpy().flatten()
    
    def next_lagged_data(self, curr_lagged_data: np.array, next_market_data: np.ndarray):
        lagged_len = len(curr_lagged_data)
        next_len = len(next_market_data)
        curr_lagged_data = np.roll(curr_lagged_data, -next_len)
        np.put(curr_lagged_data, range(lagged_len-next_len, lagged_len), next_market_data)
        return curr_lagged_data

    def get_episodic_step(self):
        return self.data_df.shape[0] 

        

    
        
        


        
            



