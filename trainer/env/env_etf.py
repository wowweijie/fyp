import gym
import os
import numpy as np
import pandas as pd
from trainer.logger import Logger 
from typing import Callable
from trainer.configs import Config


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
        start_capital=10**6,
        task_distribution=None,
    ):
        self.configs = Config.configs['env']
        self.env_name = env_name,
        self.lag = lag
        self.reward_scaling = reward_scaling
        self.price_scaling = price_scaling
        self.min_order_val = min_order_val
        self.max_order_val = max_order_val
        self.data_dir = data_dir
        self.start_capital = start_capital
        self.lower_thres=self.configs['lower_thres']
        self.upper_thres=self.configs['upper_thres']
        self.penalty_multiplier=self.configs['penalty_multiplier']
        
        # state: position, bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume 
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.logger = Logger()

        if task_distribution is not None:
            self.task_distribution = task_distribution

            # for the sake of instantiating observation dimensions etc
            self.reset_task(task_distribution[0])

        # reset
        self.position = 0
        self.open_position_val = 0
        self.step_idx = 0
        self.capital = self.start_capital
        self.asset = self.start_capital

    def sample_tasks(self, num_tasks):
        return np.random.choice(self.task_distribution, size=num_tasks, replace=True)
    
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
        self.data_df = self.data_df.dropna()
        self.data_feed_arr = self.generate_window(normalize_func=self.normalize_window)
        self.end_idx = self.data_df.shape[0] - 1
        self.bid_close_idx = self.data_df.columns.get_loc("bid_close")
        self.ask_close_idx = self.data_df.columns.get_loc("ask_close")
        market_data_dim = self.data_df.shape[1]
        # state: | position | bid ohlc, ask ohlc, dayOfWeek, timeOfDay, volume |
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=((1 + self.lag) * market_data_dim + 1,)
        )
    
    def set_task_distribution(self, *tasks):
        self.task_distribution = tasks

        # for the sake of instantiating observation dimensions etc
        self.reset_task(tasks[0])

    def step(self, actions):
        bid_close = self.data_df.iloc[self.step_idx + self.lag, self.bid_close_idx]
        ask_close = self.data_df.iloc[self.step_idx + self.lag, self.ask_close_idx]
        max_buy_avail = self.asset // ask_close 
        max_sell_avail = self.asset // bid_close
        max_buy_avail = self.upper_thres * max_buy_avail
        max_sell_avail = self.upper_thres * max_sell_avail
        target_position = self.position

        if actions[0] > self.lower_thres:
            target_position = round(actions[0] * max_buy_avail)

        elif actions[0] < -self.lower_thres:
            target_position = round(actions[0] * max_sell_avail)

        if target_position > self.position:
            buy_qty = target_position - self.position
            self.position = target_position
            self.capital -= buy_qty * ask_close

        elif target_position < self.position:
            sell_qty = self.position - target_position
            self.position = target_position
            self.capital += sell_qty * bid_close

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
        
        curr_asset = self.asset
        # self.asset is the next state asset
        self.asset = self.open_position_val + self.capital
        reward = self.asset - curr_asset
        if reward < 0:
            reward *= self.penalty_multiplier
        done = self.step_idx + self.lag == self.end_idx

        self.market_data_state = self.next_lagged_data(self.market_data_state, next_market_data)

        assert np.array_equal(self.data_feed_arr[self.step_idx],  self.normalize_window(self.market_data_state).to_numpy().flatten()), f"ERROR @ step {self.step_idx}"

        state = np.hstack((
            self.position,
            self.data_feed_arr[self.step_idx]
        ))

        if self.step_idx == 1:
            self.logger.info("STEP-1 STATE")
            self.logger.info(state)

        return state, reward, done, dict()
    
    def reset(self):
        if self.step_idx == 0:
            self.logger.info("Environment Reset")
            self.logger.info(self.__dict__)
        self.position = 0
        self.open_position_val = 0
        self.step_idx = 0
        self.capital = self.start_capital
        self.asset = self.start_capital
        self.performance = pd.DataFrame(index=np.arange(0, self.end_idx - self.lag), columns=('position', 'current_asset'))
        self.market_data_state = self.lagged_market_data(self.lag, self.lag)

        state = np.hstack((
            self.position,
            self.data_feed_arr[self.step_idx]
        ))
        
        return state
    
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
        normalized_window = sliding_window[['bid_open','bid_high','bid_low','bid_close','bid_vol',
        'ask_open','ask_high','ask_low','ask_close','ask_vol']].copy()
        mean = normalized_window.mean()
        normalized_window -= mean
        normalized_window /= mean
        normalized_window.insert(0, 'dayWeek', sliding_window['dayWeek'])
        normalized_window.insert(1, 'minute', sliding_window['minute'])
        normalized_window.insert(2, 'hour', sliding_window['hour'])
        return normalized_window

    def generate_window(self, normalize_func: Callable):
        nparray = np.zeros((len(self.data_df)-self.lag, 13*(self.lag+1)), dtype=float)
        for i in range(len(self.data_df)-self.lag):
            window_frame = self.data_df[i:i+self.lag+1]
            if normalize_func is not None:
                window_frame = self.normalize_window(window_frame)
            nparray[i] = window_frame.to_numpy().flatten()
        return nparray

    def get_episodic_step(self):
        return self.end_idx - self.lag

        

    
        
        


        
            



