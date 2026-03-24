import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from loguru import logger

class TradingEnv(gym.Env):
    """
    Custom OpenAI Gymnasium environment for Trading.
    Phase 6: RL Agent (stable-baselines3)
    State: 55 features + portfolio state
    Actions: BUY / SELL / HOLD + position size
    Reward: Profit - Risk penalty - Drawdown penalty - Transaction cost
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_balance=500000.0, transaction_cost=0.001):
        super(TradingEnv, self).__init__()

        # We need a unified dataframe sorted by time
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Action space: Continuous from -1 (Max Sell) to +1 (Max Buy)
        # Represents both direction and position sizing logic for PPO
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # State space: 55 features + balance + shares held + unrealized pnl
        # Assuming `data` passed in has exactly 55 columns
        obs_shape = self.data.shape[1] + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # Internal state tracking
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.max_net_worth = self.initial_balance
        self.net_worths = [self.initial_balance]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.max_net_worth = self.initial_balance
        self.net_worths = [self.initial_balance]

        return self._get_obs(), {}

    def _get_obs(self):
        # Current market features
        obs = self.data.iloc[self.current_step].values

        current_price = self.data.iloc[self.current_step].get('Close', 0)
        net_worth = self.balance + (self.shares_held * current_price)
        unrealized_pnl = net_worth - self.initial_balance

        # Append portfolio state
        obs = np.append(obs, [self.balance, self.shares_held, unrealized_pnl])

        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}

        current_price = self.data.iloc[self.current_step].get('Close', 1)
        prev_net_worth = self.balance + (self.shares_held * current_price)

        # Action interpretation:
        # action > 0 -> Buy (proportion of balance)
        # action < 0 -> Sell (proportion of shares held)
        action_val = action[0]
        trade_penalty = 0

        if action_val > 0.1: # Buy threshold
            # Buy `action_val` percent of current balance
            amount_to_spend = self.balance * action_val
            shares_to_buy = int(amount_to_spend / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_to_buy
                trade_penalty = cost * self.transaction_cost

        elif action_val < -0.1: # Sell threshold
            # Sell `abs(action_val)` percent of held shares
            shares_to_sell = int(self.shares_held * abs(action_val))
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held -= shares_to_sell
                trade_penalty = revenue * self.transaction_cost

        # Calculate Reward
        current_net_worth = self.balance + (self.shares_held * current_price)
        self.net_worths.append(current_net_worth)

        if current_net_worth > self.max_net_worth:
            self.max_net_worth = current_net_worth

        # Drawdown Penalty calculation
        drawdown = (self.max_net_worth - current_net_worth) / self.max_net_worth
        drawdown_penalty = drawdown * 100 # Tunable penalty weight

        # Core reward: change in net worth minus penalties
        reward = (current_net_worth - prev_net_worth) - trade_penalty - drawdown_penalty

        terminated = self.balance <= 0 or current_net_worth < self.initial_balance * 0.8 # 20% max loss

        return self._get_obs(), reward, terminated, False, {}
