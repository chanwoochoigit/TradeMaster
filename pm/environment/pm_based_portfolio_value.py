import numpy as np
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import gym

from pm.registry import ENVIRONMENT

@ENVIRONMENT.register_module()
class EnvironmentPV(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = True,
                 scaler: List[StandardScaler] = None,
                 days: int = 10,
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: int = 1e3,
                 transaction_cost_pct: float = 1e-3
                 ):
        super(EnvironmentPV, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.scaler = scaler
        self.days = days
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        if end_date is not None:
            assert end_date > start_date, "start date {}, end date {}, end date should be greater than start date".format(
                start_date, end_date)

        self.stocks = self.dataset.stocks
        self.stocks2id = self.dataset.stocks2id
        self.id2stocks = self.dataset.id2stocks
        self.aux_stocks = self.dataset.aux_stocks

        self.features_name = self.dataset.features_name
        self.prices_name = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        self.temporals_name = self.dataset.temporals_name
        self.labels_name = self.dataset.labels_name
        self.stocks_df = []

        prices = []
        if if_norm:
            print("normalize datasets")

            if self.mode == "train":
                self.scaler = []
                for df in self.dataset.stocks_df:

                    if end_date is not None:
                        df = df.loc[start_date:end_date]
                    else:
                        df = df.loc[start_date:]

                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = StandardScaler()
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.fit_transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.fit_transform(df[self.features_name])

                    self.scaler.append(scaler)
                    self.stocks_df.append(df)
            else:
                assert self.scaler is not None, "val mode or test mode is not None."

                for index, df in enumerate(self.dataset.stocks_df):

                    if end_date is not None:
                        df = df.loc[start_date:end_date]
                    else:
                        df = df.loc[start_date:]

                    df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = self.scaler[index]

                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.transform(
                            df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.transform(df[self.features_name])

                    self.stocks_df.append(df)
        else:
            print("no normalize datasets")

        self.features = []
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)

        self.prices = np.stack(prices)

        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)

        print("features shape {}, prices shape {}, labels shape {}, num days {}".format(self.features.shape,
                                                                                        self.prices.shape,
                                                                                        self.labels.shape,
                                                                                        self.features.shape[1]))

        self.num_days = self.features.shape[1]
        
        # Store training data for rolling window when in test/val mode
        self.training_features = None
        self.training_prices = None
        if self.mode in ["val", "test"] and hasattr(self.dataset, 'training_features'):
            self.training_features = self.dataset.training_features
            self.training_prices = self.dataset.training_prices

        if self.mode == "train":
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            # For test/val, start from day 0 to include ALL dates
            self.day = 0

    def get_current_date(self):
        return self.stocks_df[0].index[self.day]

    def _get_windowed_data(self, day, data_type='features'):
        """Get windowed data for the given day, using training data if needed for rolling window"""
        data = self.features if data_type == 'features' else self.prices
        
        start_idx = day - self.days + 1
        end_idx = day + 1
        
        if start_idx >= 0:
            # Normal case: all data is within current period
            return data[:, start_idx:end_idx, :]
        else:
            # Need data from before current period - use training data if available
            if self.training_features is not None and data_type == 'features':
                # Combine training data (for missing days) with current data
                training_data = self.training_features[:, start_idx:, :]  # Last |start_idx| days from training
                current_data = data[:, 0:end_idx, :]  # First end_idx days from current
                return np.concatenate([training_data, current_data], axis=1)
            elif self.training_prices is not None and data_type == 'prices':
                # Same for prices
                training_data = self.training_prices[:, start_idx:, :]
                current_data = data[:, 0:end_idx, :]
                return np.concatenate([training_data, current_data], axis=1)
            else:
                # Fallback: pad with the first available data
                needed_days = abs(start_idx)
                first_day_data = data[:, 0:1, :]  # First day data
                padding = np.tile(first_day_data, (1, needed_days, 1))  # Repeat first day
                current_data = data[:, 0:end_idx, :]
                return np.concatenate([padding, current_data], axis=1)

    def reset(self):
        if self.mode == "train":
            self.day = random.randint(self.days - 1, 3 * (self.num_days // 4))
        else:
            # For test/val, start from day 0 to include ALL dates
            self.day = 0

        state = self._get_windowed_data(self.day, 'features')

        self.state = state
        self.protfolio_value = self.initial_amount

        return state

    def get_prices(self):
        prices = self.prices[:, self.day, :]
        o, h, l, c = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3]
        return o, h, l, c

    def step(self, action: np.array = None):
        state = self.state

        weights = action.flatten()

        pre_o, pre_h, pre_l, pre_c = self.get_prices()
        
        # Get the date for the current day BEFORE incrementing (this is the day the action applies to)
        current_day_date = self.get_current_date()

        # Check if we're at the last day before incrementing
        if self.day + 1 >= self.num_days:
            done = True
            # No portfolio return calculation - episode ends
            portfolio_ret = 0.0
            # Keep current portfolio value
            reward = self.protfolio_value
        else:
            self.day = self.day + 1
            done = False
            post_o, post_h, post_l, post_c = self.get_prices()
            
            portfolio_ret = np.sum(((post_c - pre_c) / pre_c) * weights[1:])
            
            # weights[0] is the cash, weights[1:] is the stocks
            reward = (1- weights[0]) * self.protfolio_value * (1 + portfolio_ret) + \
                     weights[0] * self.protfolio_value

        self.protfolio_value = reward

        info = {
            "state": state,
            "action": action,
            "portfolio_ret": portfolio_ret,
            "portfolio_value": self.protfolio_value,
            "date": current_day_date  # Use the date BEFORE incrementing
        }

        if done:
            # For the final state, return the current state since we can't move forward
            next_state = state
        else:
            next_state = self._get_windowed_data(self.day, 'features')

        self.state = next_state

        return next_state, reward, done, info