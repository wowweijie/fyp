import gym
import os
import numpy as np
import pandas as pd
from trainer.logger import Logger 


class EtfTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

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
        self.lag = lag
        self.reward_scaling = reward_scaling
        self.price_scaling = price_scaling
        self.min_order_val = min_order_val
        self.max_order_val = max_order_val
        self.data_dir = data_dir
        self.start_capital = start_capital
        
        # state: position, bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume 
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.logger = Logger()

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
                    self.logger.info("loaded " + file)
                else:
                    self.logger.info("Unable to overwrite existing bid data")
            elif "ASK" in file:
                if df2 is None:
                    df2 = self.load_data(source + '/' + file, 'ask')
                    self.logger.info("loaded " + file)
                else:
                    self.logger.info("Unable to overwrite existing ask data")
            else:
                self.logger.info("skipping " + file)
        df = pd.concat([df1, df2], axis=1)
        df.insert(0, 'hour', df.index.hour)
        df.insert(0, 'minute', df.index.minute)
        df.insert(0, 'dayWeek', df.index.dayofweek)
        return df

    def reset_task(self, *tasks):
        self.data_df = pd.concat([self.load_task(self.data_dir + '/' + task) for task in tasks])
        self.end_idx = self.data_df.shape[0] - 1
        self.bid_close_idx = self.data_df.columns.get_loc("bid_close")
        self.ask_close_idx = self.data_df.columns.get_loc("ask_close")
        market_data_dim = self.data_df.shape[1]
        # state: | position | bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume |
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=((1 + self.lag) * market_data_dim + 1,)
        )

    def step(self, actions):
        order_val = actions[0] * self.max_order_val
        curr_asset = self.open_position_val + self.capital

        if order_val < -self.min_order_val:
            bid_close = self.data_df.iloc[self.step_idx + self.lag, self.bid_close_idx]
            if curr_asset > 0:
                sell_qty = min(curr_asset, -order_val) // bid_close
                self.capital += sell_qty * bid_close
                self.position -= sell_qty

        elif order_val > self.min_order_val:
            ask_close = self.data_df.iloc[self.step_idx + self.lag, self.ask_close_idx]
            buy_qty = min(self.capital, order_val) // ask_close
            self.position += buy_qty
            self.capital -= buy_qty * ask_close
        
        

        self.step_idx += 1
        next_market_data = self.data_df.iloc[self.step_idx + self.lag].to_numpy()
        
        # calculate wealth
        # if there is open long position
        if self.position > 0:
            # overwrite bid_close with new state 
            bid_close = next_market_data[self.bid_close_idx]
            self.open_position_val = self.position * bid_close

        # if there is open short position
        elif self.position < 0:
            # overwrite ask_close with new state 
            ask_close = next_market_data[self.ask_close_idx]
            self.open_position_val = self.position * ask_close

        # if no positions
        else:
            self.open_position_val = 0
        
        # self.asset is the next state asset
        self.asset = self.open_position_val + self.capital
        reward = self.asset - curr_asset
        done = self.step_idx + self.lag == self.end_idx

        self.market_data_state = self.next_lagged_data(self.market_data_state, next_market_data)

        state = np.hstack((
            self.position,
            self.normalize_window(self.market_data_state)
        ))

        if self.step_idx == 1:
            self.logger.info(state)

        return state, reward, done, dict()
    
    def reset(self):
        self.logger.info("Environment Reset")
        self.logger.info(self.__dict__)
        self.position = 0
        self.open_position_val = 0
        self.step_idx = 0
        self.capital = self.start_capital
        self.asset = self.start_capital
        self.performance = pd.DataFrame(index=np.arange(0, self.end_idx - self.lag), columns=('position', 'current_asset'))
        self.market_data_state = self.lagged_market_data(self.lag, self.lag)

        return np.hstack((
            self.position,
            self.normalize_window(self.market_data_state)
        ))
    
    def render(self, mode='human'):
        if self.step_idx == 0:
            self.logger.info(self.env_name + " :: Start session with rendering")

        self.performance.iloc[self.step_idx-1]['position'] = self.position
        self.performance.iloc[self.step_idx-1]['current_asset'] = self.asset
    
    def lagged_market_data(self, curr_idx: int, lag: int):
        return self.data_df.iloc[curr_idx-lag: curr_idx+1].reset_index(inplace=False, drop=True).copy()
    
    def next_lagged_data(self, curr_lagged_data: pd.DataFrame, next_market_data: pd.Series):
        curr_lagged_data = curr_lagged_data.shift(-1)
        curr_lagged_data.iloc[-1] = next_market_data
        return curr_lagged_data

    def normalize_window(self, sliding_window: pd.DataFrame):
        mean = sliding_window.mean()
        normalized_window = sliding_window.copy()
        normalized_window['bid_open'] = normalized_window['bid_open'] - mean['bid_open']
        normalized_window['bid_high'] = normalized_window['bid_high'] - mean['bid_high']
        normalized_window['bid_low'] = normalized_window['bid_low'] - mean['bid_low']
        normalized_window['bid_close'] = normalized_window['bid_close'] - mean['bid_close']

        normalized_window['ask_open'] = normalized_window['ask_open'] - mean['ask_open']
        normalized_window['ask_high'] = normalized_window['ask_high'] - mean['ask_high']
        normalized_window['ask_low'] = normalized_window['ask_low'] - mean['ask_low']
        normalized_window['ask_close'] = normalized_window['ask_close'] - mean['ask_close']
        return normalized_window.to_numpy().flatten()

    def get_episodic_step(self):
        return self.end_idx - self.lag + 1

        

    
        
        


        
            



