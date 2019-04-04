"""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import tempfile

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class DataSource:
    """
    Data source for TradingEnvironment

    Loads & preprocesses daily price & volume data
    Provides data for each new episode.
    Stocks with longest history:

    ticker  # obs
    KO      14155
    GE      14155
    BA      14155
    CAT     14155
    DIS     14155

    """

    def __init__(self, trading_days=252, ticker='AAPL', normalize=True, min_perc_days=100):
        self.ticker = ticker
        self.trading_days = trading_days + 1
        self.normalize = normalize
        self.min_perc_days = min_perc_days
        self.data = self.load_data()
        self.preprocess_data()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.idx = None

    def load_data(self):
        log.info('loading data for {}...'.format(self.ticker))
        idx = pd.IndexSlice
        with pd.HDFStore('../data/assets.h5') as store:
            df = (store['quandl/wiki/prices']
                  .loc[idx[:, self.ticker],
                       ['adj_close', 'adj_volume']]
                  .dropna())
        df.columns = ['close', 'volume']
        log.info('got data for {}...'.format(self.ticker))
        return df

    @staticmethod
    def rsi(data, window=14):
        diff = data.diff().dropna()

        up, down = diff.copy(), diff.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        rolling_up = up.rolling(window).mean()
        rolling_down = down.abs().rolling(window).mean()

        RS2 = rolling_up / rolling_down
        return 100 - (100 / (1 + RS2))

    def momentum(self, data, window=100):
        def pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        return data.rolling(window).apply(pct_rank, raw=True)

    def preprocess_data(self):
        """calculate returns and percentiles, then removes missing values"""

        # make volume positive and pre-scale
        self.data.volume = np.log(self.data.volume.replace(0, 1))

        self.data['returns'] = self.data.close.pct_change()
        self.data['close_pct_100'] = self.momentum(self.data.close, window=100)
        self.data['volume_pct_100'] = self.momentum(self.data.volume, window=100)
        self.data['close_pct_20'] = self.momentum(self.data.close, window=20)
        self.data['volume_pct_20'] = self.momentum(self.data.volume, window=20)
        self.data['return_5'] = self.data.returns.pct_change(5)
        self.data['return_21'] = self.data.returns.pct_change(21)
        self.data['rsi'] = self.rsi(self.data.close)
        self.data = self.data.replace((np.inf, -np.inf), np.nan).dropna()

        r = self.data.returns.copy()
        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)
        self.data['returns'] = r  # don't scale returns
        log.info(self.data.info())

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.trading_days
        self.idx = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.idx].values
        self.idx += 1
        self.step += 1
        done = self.step >= self.trading_days
        return obs, done


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
        based on an action and latest market return
            etc and returns the reward and a summary of the day's activity. """

        bod_position = 0.0 if self.step == 0 else self.positions[self.step - 1]
        bod_nav = 1.0 if self.step == 0 else self.navs[self.step - 1]
        bod_market_nav = 1.0 if self.step == 0 else self.market_navs[self.step - 1]

        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        self.positions[self.step] = action - 1
        self.trades[self.step] = self.positions[self.step] - bod_position

        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps
        self.costs[self.step] = trade_costs_pct + self.time_cost_bps
        reward = ((bod_position * market_return) - self.costs[self.step])
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = bod_nav * (1 + self.strategy_returns[self.step - 1])
            self.market_navs[self.step] = bod_market_nav * (1 + self.market_returns[self.step - 1])

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent from three actions:

    SHORT (0)
    FLAT (1)
    LONG (2)

    Trades cost 10bps of the change in position value.
    Going from short to long implies two trades.
    Not trading also a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode is ends with a loss.
    If the NAV hits 2.0, the agent wins.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, trading_days=252, trading_cost_bps=1e-3, time_cost_bps=1e-4, ticker='AAPL'):
        self.trading_days = trading_days
        self.ticker = ticker
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.src = DataSource(trading_days=self.trading_days, ticker=ticker)
        self.sim = TradingSimulator(steps=self.trading_days,
                                    trading_cost_bps=self.trading_cost_bps,
                                    time_cost_bps=self.time_cost_bps)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.src.min_values,
                                            self.src.max_values)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, done = self.src.take_step()
        reward, info = self.sim.take_step(action=action,
                                          market_return=observation[2])
        return observation, reward, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.src.reset()
        self.sim.reset()
        return self.src.take_step()[0]

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass

    def run_strategy(self, strategy, return_df=True):
        """Runs strategy, returns DataFrame with all steps"""
        observation = self.reset()
        done = False
        while not done:
            action = strategy(observation, self)  # call strategy
            observation, reward, done, info = self.step(action)

        return self.sim.result() if return_df else None

    def run_strategy_episodes(self, strategy, episodes=1, write_log=True, return_df=True):
        """ run provided strategy the specified # of times, possibly
            writing a log and possibly returning a dataframe summarizing activity.

            Note that writing the log is expensive and returning the df is more so.
            For training purposes, you might not want to set both.
        """
        logfile = None
        if write_log:
            logfile = tempfile.NamedTemporaryFile(delete=False, mode='w+')
            log.info('writing log to %s', logfile.name)
            need_df = write_log or return_df

        alldf = None

        for i in range(episodes):
            df = self.run_strategy(strategy, return_df=need_df)
            if write_log:
                df.to_csv(logfile, mode='ab')
                if return_df:
                    alldf = df if alldf is None else pd.concat([alldf, df], axis=0)

        return alldf
