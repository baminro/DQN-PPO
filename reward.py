import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

print("GPU:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(physical_devices))






# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import math
from numpy.random import choice
import random

from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2
import keras
from keras import layers
import numpy as np
import pandas as pd

import random
from collections import deque
import matplotlib.pylab as plt
import matplotlib.dates as mdates # Formatting dates



#Diable the warnings
import warnings
warnings.filterwarnings('ignore')


import os
directory = r'Adjusted2'
df=[]
files_in_directory = os.listdir(directory)
filtered_files = [file for file in files_in_directory if file.endswith(".csv")]
for file in filtered_files:
    path_to_file = os.path.join(directory, file)
    aux=pd.read_csv(path_to_file, encoding="utf-16")
    df.append(aux)

    """print(
        file,
        "Rows:", len(aux),
        "Max date:", aux['<DTYYYYMMDD>'].max()
    )"""

df=pd.concat(df)
df['<TICKER>']= df['<TICKER>'].str.replace('-ت', '')
df['<COL12>']= df['<COL12>'].str.replace('-ت', '')
df['<COL12>']= df['<COL12>'].str.replace('_', ' ')
df['<TICKER>']= df['<TICKER>'].str.replace('_', ' ')


df.to_excel('out.xlsx')
import datetime
from datetime import datetime as dt
df['<DTYYYYMMDD>']=df['<DTYYYYMMDD>'].apply(lambda x : dt.strptime(str(x), '%Y%m%d'))


#df['<DTYYYYMMDD>'] = pd.to_datetime(df['<DTYYYYMMDD>']).dt.date
#df['<DTYYYYMMDD>'] = df['<DTYYYYMMDD>'].dt.strftime('%Y%m%d')
df=df.set_index('<DTYYYYMMDD>')
df




df['SMAclose5']=df.groupby('<COL11>')['<CLOSE>'].transform(lambda x: x.rolling(5).mean())
df['SMAclose30']=df.groupby('<COL11>')['<CLOSE>'].transform(lambda x: x.rolling(30).mean())
df['EMAclose5']=df.groupby('<COL11>')['<CLOSE>'].transform(lambda x: x.ewm(span =5, min_periods = 5 - 1).mean())
df['EMAclose30']=df.groupby('<COL11>')['<CLOSE>'].transform(lambda x: x.ewm(span =30, min_periods = 5 - 1).mean())
MA = df['SMAclose5']
SD = df.groupby('<COL11>')['<CLOSE>'].transform(lambda x: x.rolling(window=5).std().mean())
#df['MiddleBandclose'] = MA
df['UpperBandclose'] = MA + (2 * SD)
df['LowerBandclose'] = MA - (2 * SD)

df['FIclose'] =df.groupby('<COL11>')['<CLOSE>'].transform(lambda x: x.diff(1)) * df['<VOL>']

df['dm'] = ((df['<HIGH>'] + df['<LOW>'])/2) - ((df.groupby('<COL11>')['<HIGH>'].transform(lambda x: x.shift(1)) + df.groupby('<COL11>')['<LOW>'].transform(lambda x: x.shift(1)))/2)
df['br'] = (df['<VOL>'] / 100000000) / ((df['<HIGH>'] - df['<LOW>']))
df['EMV'] = df['dm']/ df['br']
df['EMV_MAclose'] =df.groupby('<COL11>')['EMV'].transform(lambda x: x.rolling(5).mean())
del df['dm']
del df['br']


def rsi_transform(price_series, window=5):
    delta = price_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use EMA smoothing (standard for RSI)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 3. Calculate RSI by ticker using transform
df['RSIclose'] = (df.groupby('<COL11>')['<CLOSE>']
                .transform(rsi_transform, window=5))

df



df=df.reset_index()
df=df[['<DTYYYYMMDD>','<CLOSE>','<OPEN>',   '<HIGH>',   '<LOW>','<VOL>','<COL11>','<TICKER>','SMAclose5',   'SMAclose30',   'EMAclose5',    'EMAclose30',   'UpperBandclose',   'LowerBandclose',   'FIclose',  'EMV_MAclose',  'RSIclose']]
df.columns = ['Date','<CLOSE>','<OPEN>','<HIGH>',   '<LOW>','<VOL>','<COL11>','<TICKER>','SMAclose5',   'SMAclose30',   'EMAclose5',    'EMAclose30',   'UpperBandclose',   'LowerBandclose',   'FIclose',  'EMV_MAclose', 'RSIclose']
#df=df[df['Date']<='2025-05-17']


df=df.pivot(index='Date', columns='<COL11>', values=['<CLOSE>','<OPEN>',    '<HIGH>',   '<LOW>','<VOL>','SMAclose5',    'UpperBandclose',   'LowerBandclose',   'FIclose',  'EMV_MAclose',  'RSIclose'])

df=df.fillna(method='ffill')



dataset=df.dropna()
dataset

# peek at data
set_option('display.width', 100)
dataset=dataset.reset_index()
dataset['Date'] = pd.to_datetime(dataset['Date'],utc=True)
dataset['Date'] = dataset['Date'].dt.strftime('%Y-%m-%d')
dataset=dataset.set_index('Date')
#del dataset['index']
dataset.head(5)


dataindex=dataset[[('<CLOSE>', 'Overall Index')]]
dataindex
dataindex.to_csv('indexnew.csv')


dataset.to_csv('crypto_portfolionew.csv')


for ttt in range(14):

    import numpy as np
    import pandas as pd

    from IPython.core.debugger import set_trace
    from sklearn.preprocessing import MinMaxScaler

    #define a function portfolio
    def portfolio(returns, weights):
        weights = np.array(weights)
        rets = returns.mean() * 252
        retsmean=returns.mean()
        rets12=returns-retsmean
        lenx=rets12.shape[0]
        rf=0.23/252
        """for i in range(lenx):
                if (rets12.iloc[i]>0):
                    rets12.loc[:][i]=0"""

        rets12[rets12 > 0] = 0
        semicov=((rets12.T.dot(rets12))/(lenx-1))*252
        #covs = returns.cov() * 252
        P_ret = rets
        P_vol =returns.std()* np.sqrt(252)
        P_semivol =semicov* np.sqrt(252)
        P_sharpe = (P_ret-rf*252) / P_vol
        P_sortino = (P_ret-rf*252) / P_semivol
        ## 1. Calmar Ratio
        cumulative_returns = (returns + 1).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # daily → annualized compound return
        annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1
        calmar_ratio = (annualized_return-rf*252) / abs(max_drawdown) if max_drawdown != 0 else np.nan

        ## 2. Excess Return to Value-at-Risk (VaR)
        excess_return = returns.mean() - rf
        var_95 = np.percentile(returns, 5)
        excess_return_to_var = excess_return*252 / abs(var_95) if var_95 != 0 else np.nan

        # 3. Excess Return to Conditional Value-at-Risk (CVaR)
        excess_return = returns.mean() - rf
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        excess_return_to_cvar = excess_return*252 / abs(cvar_95) if cvar_95 != 0 else np.nan

        return np.array([P_ret, P_vol,P_semivol, P_sharpe,P_sortino, calmar_ratio, excess_return_to_var,excess_return_to_cvar])


    class CryptoEnvironment:

        def __init__(self, prices = 'crypto_portfolionew.csv' , capital = 1e6):
            self.prices = prices
            self.capital = capital
            self.data = self.load_data()

        def load_data(self):
            data =  pd.read_csv(self.prices)
            try:
                data.index = data['Date']
                data = data.drop(columns = ['Date'])
            except:
                multi_cols = pd.MultiIndex.from_arrays([
                         data.columns,
                         data.iloc[0],
                         data.iloc[1]])
                data.columns = multi_cols

                # Remove the first two rows since they became the new header levels
                data=data.iloc[2:].reset_index(drop=True)
                data.columns = ['_'.join(map(str, col)).strip() for col in data.columns]
                data=data.set_index('Unnamed: 0_<COL11>_Date')
                data.columns = (data.columns.str.replace(r'[\d.]+', '', regex=True).str.replace(r'nan', '', case=False, regex=True).str.replace(r'_+', '_', regex=True).str.strip('_')
    )
            return data

        def preprocess_state(self, state):

            state = state.replace([np.inf, -np.inf], np.nan)
            state = state.fillna(0)
            scaler = MinMaxScaler(feature_range=(0,1))
             # scaling dataset
            state = pd.DataFrame(scaler.fit_transform(state),index=state.index, columns=state.columns)
            return state

        def get_state(self, t, lookback, is_cov_matrix = False, is_raw_time_series = True):

            assert lookback <= t

            decision_making_state = self.data.iloc[t-lookback:t]
            decision_making_state = decision_making_state
            #set_trace()
            if is_cov_matrix:
                x = decision_making_state.cov()
                scaler =MinMaxScaler(feature_range=(0,1))
                # scaling dataset
                x =  pd.DataFrame(scaler.fit_transform(x),index=x.index, columns=x.columns)
                '''retsmean=decision_making_state.mean()
                rets12=decision_making_state-retsmean
                lenx=rets12.shape[0]
                leny=rets12.shape[1]
                for i in range(lenx):
                    for j in range(leny):
                        if (rets12.iloc[i,j]>0):
                            rets12.iloc[i,j]=0
                semicov=((rets12.T.dot(rets12))/(lenx-1))
                x=semicov'''
                return x
            elif not is_cov_matrix and not is_raw_time_series:
                x = decision_making_state.cov()
                scaler =MinMaxScaler(feature_range=(0,1))
                # scaling dataset
                #x =  scaler.fit_transform(x)
                decision_making_state = self.data.iloc[t-lookback:t]
                xx = decision_making_state.pct_change().dropna()
                #xx =  scaler.fit_transform(xx)
                xxx =  scaler.fit_transform(np.dot(xx,x))
                xxx= pd.DataFrame(xxx,index=xx.index, columns=xx.columns)

                return xxx
            else:
                if is_raw_time_series:

                    decision_making_state = self.data.iloc[t-lookback:t]
                    decision_making_state= decision_making_state.apply(pd.to_numeric, errors='coerce').drop(columns=['Unnamed: 0'], errors='ignore').dropna()
                    returnsren =decision_making_state.iloc[:, :150].pct_change()
                    cols_to_replace = decision_making_state.columns[:150]
                    decision_making_state[cols_to_replace] = returnsren[cols_to_replace].values
                    col_to_move = "<CLOSE>_Overall Index"
                    close_cols = [col for col in decision_making_state.columns if col.startswith("<CLOSE>") and col != col_to_move]
                    if close_cols:
                             last_close_index = decision_making_state.columns.get_loc(close_cols[-1])
                    else:
                             last_close_index = 0

                    cols = list(decision_making_state.columns)
                    cols.remove(col_to_move)
                    cols.insert(last_close_index + 1, col_to_move)
                    decision_making_state = decision_making_state[cols]

                return self.preprocess_state(decision_making_state.dropna())

        def get_reward(self, action, action_t, reward_t,W,capital, alpha = 0.01):
            def local_portfolio(returns, weights):
                  weights = np.array(weights)
                  rets = returns.mean()
                  retsmean = returns.mean()
                  rets12 = returns - retsmean
                  lenx = rets12.shape[0]
                  rf = 0.23 / 252

                  rets12[rets12 > 0] = 0
                  semicov = ((rets12.T.dot(rets12)) / (lenx - 1))

                  # Portfolio metrics
                  P_ret = rets
                  P_vol = returns.std()
                  P_semivol=semicov
                  P_sharpe = (P_ret - rf) / P_vol
                  P_sortino = (P_ret - rf) / P_semivol
                  # === New Metrics ===
                  ## 1. Calmar Ratio
                  cumulative_returns = (returns + 1).cumprod()
                  running_max = cumulative_returns.cummax()
                  drawdown = (cumulative_returns - running_max) / running_max
                  max_drawdown = drawdown.min()

                  # daily → annualized compound return
                  annualized_return = (cumulative_returns.iloc[-1]) ** (1 / len(cumulative_returns)) - 1
                  calmar_ratio = (annualized_return-rf) / abs(max_drawdown) if max_drawdown != 0 else np.nan

                  ## 2. Excess Return to Value-at-Risk (VaR)
                  excess_return = returns.mean() - rf
                  var_95 = np.percentile(returns, 5)
                  excess_return_to_var = excess_return / abs(var_95) if var_95 != 0 else np.nan

                  # 3. Excess Return to Conditional Value-at-Risk (CVaR)
                  excess_return = returns.mean() - rf
                  var_95 = np.percentile(returns, 5)
                  cvar_95 = returns[returns <= var_95].mean()
                  excess_return_to_cvar = excess_return / abs(cvar_95) if cvar_95 != 0 else np.nan

                  z_ret = (P_ret - np.mean(P_ret)) / (np.std(P_ret) + 1e-8)

                  z_sharpe = (P_sharpe - np.mean(P_sharpe)) / (np.std(P_sharpe) + 1e-8)

                  z_sortino = (P_sortino - np.mean(P_sortino)) / (np.std(P_sortino) + 1e-8)

                  z_var = (
                    (excess_return_to_var - np.mean(excess_return_to_var))
                    / (np.std(excess_return_to_var) + 1e-8))

                  z_cvar = (
                    (excess_return_to_cvar - np.mean(excess_return_to_cvar))
                    / (np.std(excess_return_to_cvar) + 1e-8))

                  reward1 = (
                        0.5 * z_ret
                        + 0.5 * z_sharpe)
                  reward2 = (
                        0.5 * z_ret
                        + 0.5 * z_sortino)
                  reward3 = (
                        0.5 * z_ret
                        + 0.5 * z_var)
                  reward4 = (
                        0.5 * z_ret
                        + 0.5 * z_cvar)
                  reward5 = (
                        (1/3) * z_ret
                        + (1/3) * z_cvar+ (1/3) * z_sortino)


                  reward6 = (
                        0.25 * z_ret
                        + 0.25 * z_sharpe
                        + 0.25 * z_sortino
                        + 0.25 * z_cvar)

                  return np.array([P_ret, P_vol,P_semivol, P_sharpe,P_sortino, calmar_ratio, excess_return_to_var,excess_return_to_cvar,reward1,reward2, reward3, reward4, reward5, reward6])


            # ============================================================
            # TRANSACTION COST + PORTFOLIO EVOLUTION
            # ============================================================

            # ------------------------------------------------------------
            # 1. Capital BEFORE transaction cost
            # ------------------------------------------------------------
            """print("\n========== ACTION / W DEBUG ==========")

            print("Capital:", capital)
            print("Action sum:", np.sum(action))
            print("W sum:", np.sum(W))

            print("Action:")
            print(np.round(action, 6))

            print("W:")
            print(np.round(W, 6))

            print("Turnover:")
            print(np.sum(np.abs(action - W)))

            print("======================================")"""


            capital_before_cost = float(capital)


            # ------------------------------------------------------------
            # 2. Transaction cost
            # ------------------------------------------------------------

            turnover = np.sum(np.abs(np.asarray(action) - np.asarray(W)))

            cost_rate = 0.001   # 0.1%

            transaction_cost_rate = turnover * cost_rate

            transaction_cost = (
                capital_before_cost *
                transaction_cost_rate
            )

            capital_after_cost = (
                capital_before_cost -
                transaction_cost
            )


            # ============================================================
            # 3. Portfolio allocation AFTER transaction cost
            # ============================================================

            close_columns = [
                col for col in self.data.columns
                if col.startswith('<CLOSE>_')
                and col != '<CLOSE>_Overall Index'
            ]

            data_period = (
                self.data.iloc[action_t:reward_t][close_columns]
                .ffill()
                .bfill()
            )

            data_period = data_period.apply(
                pd.to_numeric,
                errors='coerce'
            )


            # ============================================================
            # 4. Allocate remaining capital according to action
            # ============================================================

            action_array = np.asarray(action, dtype=float)

            cap_alloc = (
                capital_after_cost * action_array
            )


            # ============================================================
            # 5. Remaining capital = CASH
            # ============================================================

            initial_cash = (
                capital_after_cost -
                cap_alloc.sum()
            )

            # Numerical protection
            if abs(initial_cash) < 1e-10:
                initial_cash = 0.0


            # ============================================================
            # 6. Price evolution
            # ============================================================

            price_ratios = (
                data_period /
                data_period.shift(1)
            )

            price_ratios.iloc[0, :] = 1.0

            price_ratios = (
                price_ratios
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
            )

            cum_returns = price_ratios.cumprod()

            cum_returns.iloc[0, :] = 1.0


            # ============================================================
            # 7. Asset values through time
            # ============================================================

            assets_value = (
                cum_returns.multiply(
                    cap_alloc,
                    axis=1
                )
            )


            # ============================================================
            # 8. Stock/asset component
            # ============================================================

            hold = assets_value.sum(axis=1)


            # ============================================================
            # 9. Cash component
            # ============================================================

            rf_daily = 0.23 / 252.0

            cash_values = (
                initial_cash *
                ((1.0 + rf_daily) **
                 np.arange(len(cum_returns)))
            )

            cash_values = pd.Series(
                cash_values,
                index=cum_returns.index
            )


            # ============================================================
            # 10. Total portfolio value
            # ============================================================

            port = (
                hold +
                cash_values
            )

            port = (
                port
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
            )


            # ============================================================
            # 11. NET DAILY RETURNS
            #
            # IMPORTANT:
            # The first return is measured relative to CAPITAL BEFORE
            # transaction cost. Therefore transaction cost is included
            # in the first return of this period.
            # ============================================================

            portfolio_values = port.to_numpy(dtype=float)

            returns_array = np.zeros(
                len(portfolio_values),
                dtype=float
            )

            if len(portfolio_values) > 0:

                # First return includes transaction cost
                returns_array[0] = (
                    portfolio_values[0] /
                    capital_before_cost
                ) - 1.0


            if len(portfolio_values) > 1:

                # Subsequent daily returns
                returns_array[1:] = (
                    portfolio_values[1:] /
                    portfolio_values[:-1]
                ) - 1.0


            returns = pd.Series(
                returns_array,
                index=port.index
            )

            returns = (
                returns
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )


            # ============================================================
            # 12. FINAL CAPITAL
            # ============================================================

            capital = float(
                port.iloc[-1]
            )


            # ============================================================
            # 13. NEW PORTFOLIO WEIGHTS
            # ============================================================

            asset_weights = (
                assets_value.iloc[-1] /
                capital
            )

            asset_weights = (
                asset_weights
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            W = asset_weights.to_numpy(
                dtype=float
            )


            # ============================================================
            # 14. Portfolio reward / metrics
            # ============================================================
            
            metrics = local_portfolio(returns,W)[ttt]

            if ttt == 1 or ttt == 2:
                metrics = np.array(
                    [-metrics] * len(data_period.columns)
                )
            else:
                metrics = np.array(
                    [metrics] * len(data_period.columns)
                )
            # ============================================================
            # 15. DEBUG
            # ============================================================

            """print("\n========== TRANSACTION COST DEBUG ==========")

            print(
                "Capital before cost :",
                capital_before_cost
            )

            print(
                "Turnover            :",
                turnover
            )

            print(
                "Cost rate            :",
                cost_rate
            )

            print(
                "Effective cost rate  :",
                transaction_cost_rate
            )

            print(
                "Transaction cost     :",
                transaction_cost
            )

            print(
                "Capital after cost   :",
                capital_after_cost
            )

            print(
                "Initial cash         :",
                initial_cash
            )

            print(
                "Final capital        :",
                capital
            )

            print(
                "First net return     :",
                returns.iloc[0]
            )

            print(
                "Cumulative return    :",
                (1.0 + returns).prod() - 1.0
            )

            print(
                "New weights sum      :",
                W.sum()
            )

            print("============================================")"""


            # ============================================================
            # 16. RETURN
            # ============================================================

            return (
                returns,
                metrics,
                W,
                capital
            )
            #sharpe = np.array([sharpe] * (len(self.data.columns)-311))

            #rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]

            # === New Metrics ===
            ## 1. Calmar Ratio
            """cumulative_returns = kll['port'] / kll['port'].iloc[0]
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            annualized_return = (cumulative_returns.iloc[-1]) ** (1 / len(cumulative_returns)) - 1
            sharpe = (annualized_return - rf)/ abs(max_drawdown) if max_drawdown != 0 else np.nan
            sharpe = np.array([sharpe] * (len(self.data.columns)-191))

            ## 2. Excess Return to Value-at-Risk (VaR)
            excess_return = returns.mean() - rf
            var_95 = np.percentile(returns, 5)
            sharpe = excess_return / abs(var_95) if var_95 != 0 else np.nan
            sharpe = np.array([sharpe] * (len(self.data.columns)-191))"""

            #return returns, sharpe, W, capital,




    import numpy as np
    import tensorflow as tf
    import keras.backend as K
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D, Dense, Dropout, Flatten, LSTM, LayerNormalization, MultiHeadAttention, Add
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Permute, Softmax, Multiply
    from tensorflow.keras.layers import Dense, Multiply, Lambda,Embedding
    from tensorflow.keras.layers import (
        Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,
        LayerNormalization, MultiHeadAttention, Add
    )
    from tensorflow.keras.models import Model
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    import numpy as np
    from tensorflow.keras.layers import Layer



    """class Attention(tf.keras.layers.Layer):
              def __init__(self, units):
                  super(Attention, self).__init__()
                  self.W = tf.keras.layers.Dense(units)
                  self.V = tf.keras.layers.Dense(1)
              def call(self, inputs):
                  # Compute attention scores
                  score = tf.nn.tanh(self.W(inputs))
                  attention_weights = tf.nn.softmax(self.V(score), axis=1)

                  # Apply attention weights to input
                  context_vector = attention_weights * inputs
                  context_vector = tf.reduce_sum(context_vector, axis=1)

                  return context_vector"""

    class Agent:

        def __init__(
                         self,
                         portfolio_size,
                         is_eval = False,
                         allow_short = False,
                     ):

            self.portfolio_size = portfolio_size
            self.allow_short = allow_short
            self.input_shape = (62,341, )
            self.action_size = 3 # sit, buy, sell

            self.memory4replay = []
            self.is_eval = is_eval

            self.alpha = 0.1
            self.gamma = 0.95
            self.epsilon = 1
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.model = self._model()

        def _model(self):
                inputs = Input(shape=self.input_shape)

                x = Conv1D(32, kernel_size=2, activation="relu", padding="same")(inputs)
                x = MaxPooling1D(pool_size=2)(x)

                x = Conv1D(32, kernel_size=2, activation="relu", padding="same")(x)
                x = MaxPooling1D(pool_size=2)(x)

                x = Flatten()(x)
                x = Dense(32, activation='elu')(x)
                x = Dropout(0.4)(x)

                predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-311)]
                model = Model(inputs=inputs, outputs=predictions)
                model.compile(optimizer='adam', loss='mse')
                return model

        """def _model(self):
                inputs = Input(shape=self.input_shape)

                x = Conv1D(64, kernel_size=2, activation="relu", padding="same")(inputs)
                x = MaxPooling1D(pool_size=2)(x)

                x = LSTM(64, return_sequences=False)(x)
                x = Dense(64, activation='elu')(x)
                x = Dropout(0.4)(x)

                predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-311)]
                model = Model(inputs=inputs, outputs=predictions)
                model.compile(optimizer='adam', loss='mse')
                return model

        def _transformer_encoder(self, x, head_size=4, num_heads=2, ff_dim=64, dropout=0.4):
                attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
                attn_output = Dropout(dropout)(attn_output)
                x = Add()([x, attn_output])
                x = LayerNormalization(epsilon=1e-6)(x)

                ffn = Dense(ff_dim, activation="relu")(x)
                ffn = Dense(x.shape[-1])(ffn)
                x = Add()([x, ffn])
                x = LayerNormalization(epsilon=1e-6)(x)
                return x

        class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
                def __init__(self, max_len, d_model):
                    super().__init__()
                    pos = np.arange(max_len)[:, np.newaxis]
                    i = np.arange(d_model)[np.newaxis, :]
                    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
                    angle_rads = pos * angle_rates
                    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
                    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
                    self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

                def call(self, x):
                    seq_len = tf.shape(x)[1]
                    return x + self.pos_encoding[:, :seq_len, :]

        def _model(self):
                inputs = Input(shape=self.input_shape)

                x = Dense(64)(inputs)  # Project features to d_model dimension
                x = self.SinusoidalPositionalEncoding(62, 64)(x)
                x = self._transformer_encoder(x)

                x = GlobalMaxPooling1D()(x)
                x = Dense(64, activation='elu')(x)
                x = Dropout(0.4)(x)

                predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-311)]
                model = Model(inputs=inputs, outputs=predictions)
                model.compile(optimizer='adam', loss='mse')
                return model"""

        """def attention_block(self,x):
           # x: (batch, timesteps, features)
           attention = Dense(1, activation='tanh')(x)  # (batch, timesteps, 1)
           attention = Lambda(lambda a: tf.nn.softmax(a, axis=1))(attention)  # softmax over time
           x = Multiply()([x, attention])  # (batch, timesteps, features)
           return x

        def _model(self):
            inputs = Input(shape=self.input_shape)

            x = Conv1D(filters=128, kernel_size=2, activation='relu', padding='valid')(inputs)
            x = MaxPooling1D(pool_size=2, strides=1)(x)
            x = self.attention_block(x)

            x = Flatten()(x)
            x = Dense(128, activation='elu')(x)
            x = Dropout(0.5)(x)
            x = Dense(64, activation='elu')(x)
            x = Dropout(0.5)(x)

            predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-191)]

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam', loss='mse')
            return model"""








        """def _model(self):

            inputs = Input(shape=self.input_shape)
            #Flatten()
            #x = Flatten()(inputs)
            x=keras.layers.Conv1D(filters=252, kernel_size=2,
                          strides=1, padding="valid",
                          activation="relu")(inputs)
            #x =Flatten()(x)

            x=keras.layers.MaxPooling1D(pool_size=2,strides=1, padding="valid")(x)
            x=keras.layers.Conv1D(filters=252, kernel_size=2,
                          strides=1, padding="valid",
                          activation="relu")( inputs)
            x=keras.layers.MaxPooling1D(pool_size=2,strides=1, padding="valid")(x)
            x=keras.layers.Conv1D(filters=252, kernel_size=2,
                          strides=1, padding="valid",
                          activation="relu")( inputs)
            x=keras.layers.MaxPooling1D(pool_size=2,strides=1, padding="valid")(x)


            #x=keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True),
                                  # backward_layer=tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, go_backwards=True),
                                  #merge_mode="concat")(x)
            #x=keras.layers.Conv1D(filters=32, kernel_size=2,
                          #strides=1, padding="valid",
                          #activation="relu")(x)
           # x=keras.layers.MaxPooling1D(pool_size=2,strides=2, padding="valid")(x)
            #Attention(64),
            #tf.keras.layers.Reshape((128,1)),
            #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(126, return_sequences=True),
                                     #backward_layer=tf.keras.layers.LSTM(126, activation='relu', return_sequences=True, go_backwards=True),
                                     #merge_mode="concat"),
            x = Dense(256, activation='elu')(x)

            #
            #x=keras.layers.LSTM(256,activation='relu',return_sequences=False)(x)
            #x=keras.layers.Reshape((256,1))(x)
            #x=keras.layers.LSTM(30,activation='relu',return_sequences=False)(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='elu')(x)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            predictions = []
            for i in range(self.portfolio_size-191):
                asset_dense = Dense(self.action_size, activation='linear')(x)
                predictions.append(asset_dense)

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam', loss='mse')
            return model"""

        """def _model(self):
               inputs = Input(shape=self.input_shape)

               x = Conv1D(filters=128, kernel_size=2, activation='relu', padding='valid')(inputs)
               x = MaxPooling1D(pool_size=2, strides=1)(x)
               x = LSTM(128, return_sequences=False)(x)

               x = Dense(128, activation='elu')(x)
               x = Dropout(0.5)(x)
               x = Dense(64, activation='elu')(x)
               x = Dropout(0.5)(x)

               x = Flatten()(x)
               predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-191)]

               model = Model(inputs=inputs, outputs=predictions)
               model.compile(optimizer='adam', loss='mse')
               return model"""


        """def _transformer_encoder(self, x, head_size=4, num_heads=4, ff_dim=8, dropout=0.5):
            # Multi-head attention block
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
            attn_output = Dropout(dropout)(attn_output)
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward block
            ffn = Dense(ff_dim, activation="relu")(x)
            ffn = Dense(x.shape[-1])(ffn)
            x = Add()([x, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)
            return x


        def _model(self):
            inputs = Input(shape=self.input_shape)
            x = Conv1D(32, 3, activation="relu", padding="same")(inputs)
            x = Conv1D(16, 3, activation="relu", padding="same")(x)

            x = self._transformer_encoder(x)  # encoder with default config
            x = GlobalMaxPooling1D()(x)
            x = Flatten()(x)
            x = Dense(128, activation='elu')(x)
            x = Dropout(0.4)(x)
            x = Dense(64, activation='elu')(x)
            x = Dropout(0.4)(x)

            predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size - 191)]

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam', loss='mse')
            return model"""





        """def attention_block(self,x):
           # x: (batch, timesteps, features)
           attention = Dense(1, activation='tanh')(x)  # (batch, timesteps, 1)
           attention = Lambda(lambda a: tf.nn.softmax(a, axis=1))(attention)  # softmax over time
           x = Multiply()([x, attention])  # (batch, timesteps, features)
           return x

        def _model(self):
            inputs = Input(shape=self.input_shape)

            x = Conv1D(filters=128, kernel_size=2, activation='relu', padding='valid')(inputs)
            x = MaxPooling1D(pool_size=2, strides=1)(x)
            x = self.attention_block(x)

            x = Flatten()(x)
            x = Dense(128, activation='elu')(x)
            x = Dropout(0.5)(x)
            x = Dense(64, activation='elu')(x)
            x = Dropout(0.5)(x)

            predictions = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-191)]

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam', loss='mse')
            return model"""


        """class SinusoidalPositionalEncoding(Layer):
            def __init__(self, max_len, d_model):
                super().__init__()
                self.max_len = max_len
                self.d_model = d_model

                # Create positional encodings once at initialization
                pos = np.arange(max_len)[:, np.newaxis]
                i = np.arange(d_model)[np.newaxis, :]
                angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
                angle_rads = pos * angle_rates

                # Apply sin to even indices and cos to odd indices
                angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
                angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

                # Convert to constant tensor
                self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)  # shape (1, max_len, d_model)

            def call(self, x):
                seq_len = tf.shape(x)[1]
                return x + self.pos_encoding[:, :seq_len, :]

        class PositionalEmbedding(Layer):
            def __init__(self, max_len, d_model):
                super().__init__()
                self.pos_embedding = Embedding(input_dim=max_len, output_dim=d_model)

            def call(self, x):
                positions = tf.range(start=0, limit=tf.shape(x)[1])
                pos_encoding = self.pos_embedding(positions)
                return x + pos_encoding

        def _transformer_encoder(self, x, head_size=16, num_heads=4, ff_dim=64, dropout=0.2):
            # Multi-head attention block
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
            attn_output = Dropout(dropout)(attn_output)
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward block
            ffn = Dense(ff_dim, activation="relu")(x)
            ffn = Dense(x.shape[-1])(ffn)
            x = Add()([x, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)
            return x

        def _model(self):
            inputs = Input(shape=self.input_shape)  # (timesteps, features)

            # 1. CNN
            x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(inputs)
            x = MaxPooling1D(pool_size=2, strides=1)(x)

            # 2. LSTM
            x = LSTM(16, return_sequences=True)(x)


            sequence_length = self.input_shape[0]
            d_model = x.shape[-1]
            positions = tf.expand_dims(tf.range(start=0, limit=sequence_length, delta=1), 0)  # shape (1, time)
            pos_encoding = Embedding(input_dim=1000, output_dim=d_model)(positions)           # shape (1, time, d_model)
            x = x + pos_encoding

            #x = self.PositionalEmbedding(max_len=1000, d_model=x.shape[-1])(x)
            x =self.SinusoidalPositionalEncoding(max_len=1000, d_model=x.shape[-1])(x)


            # 3. Transformer encoder
            x = self._transformer_encoder(x)

            # 4. Dense layers
            x = Flatten()(x)
            x = Dense(32, activation='elu')(x)
            x = Dropout(0.2)(x)
            x = Dense(16, activation='elu')(x)
            x = Dropout(0.2)(x)

            # 5. Multi-head output (one head per asset)
            outputs = [Dense(self.action_size, activation='linear')(x) for _ in range(self.portfolio_size-191)]

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse')
            return model"""



        def nn_pred_to_weights(self, pred, allow_short = False):

            weights = np.zeros(len(pred))
            raw_weights = np.argmax(pred, axis=-1)

            saved_min = None

            for e, r in enumerate(raw_weights):
                if r == 0: # sit
                    weights[e] = 0
                elif r == 1: # buy
                    weights[e] = np.abs(pred[e][0][r])
                else:
                    weights[e] = -np.abs(pred[e][0][r])
            #sum of absolute values in short is allowed
            if not allow_short:
                weights += np.abs(np.min(weights))
                saved_min = np.abs(np.min(weights))
                saved_sum = np.sum(weights)
            else:
                saved_sum = np.sum(np.abs(weights))

            weights /= saved_sum
            return weights, saved_min, saved_sum
        #return the action based on the state, uses the NN function
        def act(self, state):

            if not self.is_eval and random.random() <= self.epsilon:
                w = np.random.normal(0, 1, size = (self.portfolio_size-311, ))

                saved_min = None

                if not self.allow_short:
                    w += np.abs(np.min(w))
                    saved_min = np.abs(np.min(w))

                saved_sum = np.sum(w)
                w /= saved_sum
                return w , saved_min, saved_sum
            pred = self.model.predict(np.expand_dims(state.values, 0))
            return self.nn_pred_to_weights(pred, self.allow_short)

        def expReplay(self, batch_size):

            def weights_to_nn_preds_with_reward(action_weights,
                                                reward,
                                                Q_star = np.zeros((self.portfolio_size-311, self.action_size))):

                Q = np.zeros((self.portfolio_size-311, self.action_size))
                for i in range(self.portfolio_size-311):
                    if action_weights[i] == 0:
                        Q[i][0] = reward[i] + self.gamma * np.max(Q_star[i][0])
                    elif action_weights[i] > 0:
                        Q[i][1] = reward[i] + self.gamma * np.max(Q_star[i][1])
                    else:
                        Q[i][2] = reward[i] + self.gamma * np.max(Q_star[i][2])
                return Q

            def restore_Q_from_weights_and_stats(action):
                action_weights, action_min, action_sum = action[0], action[1], action[2]
                action_weights = action_weights * action_sum
                if action_min != None:
                    action_weights = action_weights - action_min
                return action_weights

            for (s, s_, action, reward, done) in self.memory4replay:

                action_weights = restore_Q_from_weights_and_stats(action)
                #Reward =reward if not in the terminal state.
                Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward)
                s, s_ = s.values, s_.values

                if not done:
                    # reward + gamma * Q^*(s_, a_)
                    Q_star = self.model.predict(np.expand_dims(s_, 0))
                    Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward, np.squeeze(Q_star))

                Q_learned_value = [xi.reshape(1, -1) for xi in Q_learned_value]
                Q_current_value = self.model.predict(np.expand_dims(s, 0))
                Q = [np.add(a * (1-self.alpha), q * self.alpha) for a, q in zip(Q_current_value, Q_learned_value)]

                # update current Q function with new optimal value

                self.model.fit(np.expand_dims(s, 0), Q, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                Q_learned_value = [xi.reshape(1, -1) for xi in Q_learned_value]
                Q_current_value = self.model.predict(np.expand_dims(s, 0))
                Q = [np.add(a * (1-self.alpha), q * self.alpha) for a, q in zip(Q_current_value, Q_learned_value)]

                # update current Q function with new optimal value

                self.model.fit(np.expand_dims(s, 0), Q, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay







    import numpy as np
    import tensorflow as tf

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        MaxPooling1D,
        Flatten,
        Dense
    )
    from tensorflow.keras.optimizers import Adam


    class PPOAgent:

        def __init__(
            self,
            portfolio_size,
            input_shape=(62, 341),
            gamma=0.95,
            gae_lambda=0.95,
            clip_ratio=0.20,
            actor_lr=1e-4,
            critic_lr=1e-4
        ):

            self.portfolio_size = portfolio_size

            # 341 - 311 = 30 assets
            self.n_assets = portfolio_size - 311

            self.input_shape = input_shape

            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_ratio = clip_ratio

            # PPO memory
            self.memory = []

            # Actor
            self.actor = self._build_actor()

            # Critic
            self.critic = self._build_critic()

            self.actor_optimizer = Adam(
                learning_rate=actor_lr
            )

            self.critic_optimizer = Adam(
                learning_rate=critic_lr
            )


        # ==========================================================
        # CNN FEATURE EXTRACTOR
        # ==========================================================

        def _cnn_features(self):

            inputs = Input(
                shape=self.input_shape
            )

            x = Conv1D(
                32,
                kernel_size=2,
                activation="relu",
                padding="same"
            )(inputs)

            x = MaxPooling1D(
                pool_size=2
            )(x)

            x = Conv1D(
                32,
                kernel_size=2,
                activation="relu",
                padding="same"
            )(x)

            x = MaxPooling1D(
                pool_size=2
            )(x)

            x = Flatten()(x)

            x = Dense(
                32,
                activation="elu"
            )(x)

            # ------------------------------------------------------
            # IMPORTANT:
            # Dropout removed.
            #
            # PPO must evaluate the policy consistently between:
            # data collection and policy update.
            # ------------------------------------------------------

            return inputs, x


        # ==========================================================
        # ACTOR
        # ==========================================================

        def _build_actor(self):

            inputs, x = self._cnn_features()

            # Mean of Gaussian policy
            mu = Dense(
                self.n_assets,
                activation="linear",
                name="mu"
            )(x)

            # Log standard deviation
            log_std = Dense(
                self.n_assets,
                activation="linear",
                name="log_std"
            )(x)

            return Model(
                inputs,
                [mu, log_std]
            )


        # ==========================================================
        # CRITIC
        # ==========================================================

        def _build_critic(self):

            inputs, x = self._cnn_features()

            value = Dense(
                1,
                activation="linear"
            )(x)

            return Model(
                inputs,
                value
            )


        # ==========================================================
        # GET POLICY DISTRIBUTION
        # ==========================================================

        def _policy_distribution(self, state):

            state_array = np.expand_dims(
                state.values.astype(np.float32),
                axis=0
            )

            mu, log_std = self.actor(
                state_array,
                training=False
            )

            mu = mu.numpy()[0]

            log_std = log_std.numpy()[0]

            # Numerical protection
            log_std = np.clip(
                log_std,
                -5.0,
                1.0
            )

            std = np.exp(log_std)

            return mu, std


        # ==========================================================
        # ACTION
        # ==========================================================

        def act(
            self,
            state,
            deterministic=False
        ):

            mu, std = self._policy_distribution(
                state
            )

            # ------------------------------------------------------
            # Generate Gaussian action
            # ------------------------------------------------------

            if deterministic:

                sampled_action = mu.copy()

            else:

                sampled_action = (
                    mu
                    +
                    std *
                    np.random.normal(
                        size=self.n_assets
                    )
                )

            sampled_action = np.asarray(
                sampled_action,
                dtype=np.float32
            )

            sampled_action = np.nan_to_num(
                sampled_action,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # ------------------------------------------------------
            # Convert raw action to portfolio weights
            #
            # IMPORTANT:
            # We DO NOT modify sampled_action.
            #
            # sampled_action is used for PPO log probability.
            #
            # logits is used only for softmax.
            # ------------------------------------------------------

            logits = (
                sampled_action
                -
                np.max(sampled_action)
            )

            exp_action = np.exp(
                logits
            )

            weights = (
                exp_action
                /
                (np.sum(exp_action) + 1e-8)
            )

            # Numerical protection
            weights = np.maximum(
                weights,
                0.0
            )

            weights /= (
                np.sum(weights)
                +
                1e-8
            )

            # ------------------------------------------------------
            # Value estimate
            # ------------------------------------------------------

            state_array = np.expand_dims(
                state.values.astype(np.float32),
                axis=0
            )

            value = self.critic(
                state_array,
                training=False
            ).numpy()[0, 0]

            # ------------------------------------------------------
            # IMPORTANT:
            # Log probability is calculated using the ORIGINAL
            # sampled Gaussian action.
            # ------------------------------------------------------

            log_prob = self.log_probability(
                state,
                sampled_action
            )

            return (
                weights,
                value,
                log_prob,
                sampled_action
            )


        # ==========================================================
        # LOG PROBABILITY
        # ==========================================================

        def log_probability(
            self,
            state,
            raw_action
        ):

            mu, std = self._policy_distribution(
                state
            )

            var = std ** 2

            log_prob = (
                -0.5
                *
                (
                    (
                        (raw_action - mu) ** 2
                    )
                    /
                    var
                    +
                    2.0 * np.log(std)
                    +
                    np.log(2.0 * np.pi)
                )
            )

            return float(
                np.sum(log_prob)
            )


        # ==========================================================
        # STORE EXPERIENCE
        # ==========================================================

        def remember(
            self,
            state,
            action,
            reward,
            value,
            log_prob,
            done
        ):

            self.memory.append(
                (
                    state.values.astype(
                        np.float32
                    ),

                    np.asarray(
                        action,
                        dtype=np.float32
                    ),

                    float(reward),

                    float(value),

                    float(log_prob),

                    bool(done)
                )
            )


        # ==========================================================
        # COMPUTE RETURNS AND ADVANTAGES
        # ==========================================================

        def compute_advantages(self):

            rewards = np.asarray(
                [x[2] for x in self.memory],
                dtype=np.float32
            )

            values = np.asarray(
                [x[3] for x in self.memory],
                dtype=np.float32
            )

            dones = np.asarray(
                [x[5] for x in self.memory],
                dtype=np.float32
            )

            advantages = np.zeros_like(
                rewards,
                dtype=np.float32
            )

            gae = 0.0

            for t in reversed(range(len(rewards))):

                # Value of next state
                if t == len(rewards) - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]

                # TD residual
                delta = (
                    rewards[t]
                    + self.gamma * next_value * (1.0 - dones[t])
                    - values[t]
                )

                # GAE
                gae = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * (1.0 - dones[t])
                    * gae
                )

                advantages[t] = gae

            # Returns used for critic training
            returns = advantages + values

            # Normalize advantages
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std() + 1e-8
            )

            return (
                returns.astype(np.float32),
                advantages.astype(np.float32)
            )


        # ==========================================================
        # TRAIN PPO
        # ==========================================================

        def train(
            self,
            epochs=50
        ):

            if len(self.memory) == 0:

                return

            # ------------------------------------------------------
            # Convert memory to arrays
            # ------------------------------------------------------

            states = np.asarray(
                [
                    x[0]
                    for x in self.memory
                ],
                dtype=np.float32
            )

            actions = np.asarray(
                [
                    x[1]
                    for x in self.memory
                ],
                dtype=np.float32
            )

            old_log_probs = np.asarray(
                [
                    x[4]
                    for x in self.memory
                ],
                dtype=np.float32
            )

            # ------------------------------------------------------
            # Returns and advantages
            # ------------------------------------------------------

            returns, advantages = (
                self.compute_advantages()
            )

            returns_tf = tf.convert_to_tensor(
                returns,
                dtype=tf.float32
            )

            advantages_tf = tf.convert_to_tensor(
                advantages,
                dtype=tf.float32
            )

            old_log_probs_tf = tf.convert_to_tensor(
                old_log_probs,
                dtype=tf.float32
            )


            # ======================================================
            # PPO UPDATE
            # ======================================================

            for _ in range(epochs):

                # --------------------------------------------------
                # ACTOR UPDATE
                # --------------------------------------------------

                with tf.GradientTape() as actor_tape:

                    mu, log_std = self.actor(
                        states,
                        training=True
                    )

                    log_std = tf.clip_by_value(
                        log_std,
                        -5.0,
                        1.0
                    )

                    std = tf.exp(
                        log_std
                    )

                    var = std ** 2

                    # Gaussian log probability
                    log_probs = (
                        -0.5
                        *
                        (
                            (
                                (
                                    actions
                                    -
                                    mu
                                ) ** 2
                            )
                            /
                            var
                        )
                        +
                        (-log_std)
                        -
                        0.5 * np.log(
                            2.0 * np.pi
                        )
                    )

                    log_probs = tf.reduce_sum(
                        log_probs,
                        axis=1
                    )

                    # --------------------------------------------------
                    # PPO probability ratio
                    # --------------------------------------------------

                    ratio = tf.exp(
                        log_probs
                        -
                        old_log_probs_tf
                    )

                    clipped_ratio = tf.clip_by_value(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio
                    )

                    surrogate1 = (
                        ratio
                        *
                        advantages_tf
                    )

                    surrogate2 = (
                        clipped_ratio
                        *
                        advantages_tf
                    )

                    actor_loss = (
                        -tf.reduce_mean(
                            tf.minimum(
                                surrogate1,
                                surrogate2
                            )
                        )
                    )


                actor_grads = actor_tape.gradient(
                    actor_loss,
                    self.actor.trainable_variables
                )

                self.actor_optimizer.apply_gradients(
                    zip(
                        actor_grads,
                        self.actor.trainable_variables
                    )
                )


                # --------------------------------------------------
                # CRITIC UPDATE
                # --------------------------------------------------

                with tf.GradientTape() as critic_tape:

                    values_pred = tf.squeeze(
                        self.critic(
                            states,
                            training=True
                        ),
                        axis=1
                    )

                    critic_loss = tf.reduce_mean(
                        tf.square(
                            returns_tf
                            -
                            values_pred
                        )
                    )


                critic_grads = critic_tape.gradient(
                    critic_loss,
                    self.critic.trainable_variables
                )

                self.critic_optimizer.apply_gradients(
                    zip(
                        critic_grads,
                        self.critic.trainable_variables
                    )
                )


            # ------------------------------------------------------
            # Clear PPO memory after update
            # ------------------------------------------------------

            self.memory = []


        # ==========================================================
        # SAVE
        # ==========================================================

        def save(
            self,
            path
        ):

            self.actor.save(
                path + "_actor.keras"
            )

            self.critic.save(
                path + "_critic.keras"
            )









    N_ASSETS = 341 #53
    agent = Agent(N_ASSETS)

    ppo_agent = PPOAgent( portfolio_size=N_ASSETS, input_shape=(62, 341), gamma=0.95, gae_lambda=0.95, clip_ratio=0.20, actor_lr=7e-5, critic_lr=1e-4 )

    env = CryptoEnvironment()


    window_size = 63
    episode_count = 50
    batch_size = 32
    rebalance_period = 21 #every 90 days weight change


    import numpy as np

    class EarlyStoppingRL:
        def __init__(self, monitor='reward', mode='max', patience=10, min_delta=5e-5, verbose=True):
            """
            Early stopping callback for reinforcement learning training loops.

            Parameters:
            - monitor (str): Metric name to track (e.g., 'reward', 'sharpe')
            - mode (str): 'max' (for reward/sharpe) or 'min' (for loss)
            - patience (int): Episodes to wait for improvement before stopping
            - min_delta (float): Minimum change to qualify as improvement
            - verbose (bool): Print messages during training
            """
            self.monitor = monitor
            self.mode = mode
            self.patience = patience
            self.min_delta = min_delta
            self.verbose = verbose

            self.best_score = -np.inf if mode == 'max' else np.inf
            self.wait = 0
            self.stopped_epoch = 0
            self.stop_training = False
            self.best_episode = 0

        def check(self, current_value, episode):
            """
            Check if training should stop.

            Parameters:
            - current_value (float): The current metric value (e.g., average reward)
            - episode (int): The current episode number

            Returns:
            - stop_training (bool): Whether to stop training
            """
            if self.mode == 'max':
                improvement = current_value - self.best_score
            else:
                improvement = self.best_score - current_value

            if improvement > self.min_delta:
                self.best_score = current_value
                self.best_episode = episode
                self.wait = 0
                if self.verbose:
                    print(f"→ {self.monitor} improved to {current_value:.4f} at episode {episode}")
            else:
                self.wait += 1
                if self.verbose:
                    print(f"No significant improvement in {self.monitor} at episode {episode} (wait={self.wait})")

                if self.wait >= self.patience:
                    self.stop_training = True
                    self.stopped_epoch = episode
                    if self.verbose:
                        print(f"✋ Early stopping triggered at episode {episode}. Best {self.monitor} = {self.best_score:.4f} at episode {self.best_episode}")

            return self.stop_training




    import numpy as np
    from scipy.optimize import minimize


    def minimum_variance_weights(returns_window):
        """
        Minimum Variance Portfolio
        Long-only, fully invested
        """

        returns_window = np.asarray(returns_window, dtype=float)

        # Covariance matrix
        cov = np.cov(returns_window, rowvar=False)

        n_assets = cov.shape[0]

        # Initial equal weights
        x0 = np.ones(n_assets) / n_assets

        # Portfolio variance
        def portfolio_variance(w):
            return w.T @ cov @ w

        # Fully invested
        constraints = [
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            }
        ]

        # Long-only
        bounds = [(0.0, 1.0)] * n_assets

        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000,
                'ftol': 1e-10
            }
        )

        if not result.success:
            print("Minimum Variance optimization failed:",
                  result.message)
            return x0

        return result.x



    def markowitz_weights(
        returns_window,
        risk_aversion=1.0
    ):
        """
        Long-only Markowitz Mean-Variance Portfolio
        """

        returns_window = np.asarray(
            returns_window,
            dtype=float
        )

        mu = np.mean(
            returns_window,
            axis=0
        )

        cov = np.cov(
            returns_window,
            rowvar=False
        )

        n_assets = len(mu)

        x0 = np.ones(n_assets) / n_assets

        def objective(w):

            expected_return = np.dot(mu, w)

            variance = w.T @ cov @ w

            utility = (
                expected_return
                - 0.5 * risk_aversion * variance
            )

            return -utility

        constraints = [
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            }
        ]

        bounds = [(0.0, 1.0)] * n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000,
                'ftol': 1e-10
            }
        )

        if not result.success:
            print(
                "Markowitz optimization failed:",
                result.message
            )
            return x0

        return result.x


    def risk_parity_weights(returns_window):

        returns_window = np.asarray(
            returns_window,
            dtype=float
        )

        cov = np.cov(
            returns_window,
            rowvar=False
        )

        n_assets = cov.shape[0]

        x0 = np.ones(n_assets) / n_assets

        def objective(w):

            portfolio_vol = np.sqrt(
                w.T @ cov @ w
            )

            marginal_risk = cov @ w

            risk_contribution = (
                w * marginal_risk
            ) / portfolio_vol

            target = portfolio_vol / n_assets

            return np.sum(
                (risk_contribution - target) ** 2
            )

        constraints = [
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            }
        ]

        bounds = [(0.0, 1.0)] * n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 2000,
                'ftol': 1e-10
            }
        )

        if not result.success:
            return x0

        return result.x

    def max_sharpe_weights(
        returns_window,
        rf=0.23/252
    ):

        returns_window = np.asarray(
            returns_window,
            dtype=float
        )

        mu = np.mean(
            returns_window,
            axis=0
        )

        cov = np.cov(
            returns_window,
            rowvar=False
        )

        n_assets = len(mu)

        x0 = np.ones(n_assets) / n_assets

        def objective(w):

            portfolio_return = np.dot(mu, w)

            portfolio_vol = np.sqrt(
                w.T @ cov @ w
            )

            sharpe = (
                portfolio_return - rf
            ) / (portfolio_vol + 1e-12)

            return -sharpe

        constraints = [
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            }
        ]

        bounds = [(0.0, 1.0)] * n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000,
                'ftol': 1e-10
            }
        )

        if not result.success:
            return x0

        return result.x




    early_stop = EarlyStoppingRL(monitor='reward', mode='max', patience=10, min_delta=7e-5)
    early_stop_dqn = EarlyStoppingRL(
        monitor='reward',
        mode='max',
        patience=10,
        min_delta=7e-5
    )

    early_stop_ppo = EarlyStoppingRL(
        monitor='reward',
        mode='max',
        patience=10,
        min_delta=7e-5
    )

    reward_list_dqn = []
    reward_list_ppo = []


    reward_list = []  # store average reward per episode
    max_episodes = episode_count

    for e in range(max_episodes):

        agent.is_eval = False
        data_length = len(env.data)

        returns_history = []
        returns_history_equal = []
        returns_history_minvar = []

        rewards_history = []
        equal_rewards = []
        minvar_rewards = []

        actions_to_show = []
        actions_to_show1 = []
        diffc = []

        print(f"Episode {e + 1}/{max_episodes}, epsilon = {agent.epsilon:.4f}")

        s = env.get_state(np.random.randint(window_size + 1, data_length - 505 - window_size - 1), window_size)
        total_profit = 0
        W0 = [0. for _ in range(N_ASSETS - 311)]
        W0eq = [0. for _ in range(N_ASSETS - 311)]
        W0minvar = [0. for _ in range(N_ASSETS - 311)]

        initial_capital = float(1000)
        initial_capitaleq = float(1000)
        initial_capital_minvar = float(1000)


        ppo_agent.memory = [] 
        W0ppo = [ 0.0 for _ in range(N_ASSETS - 311) ] 
        initial_capitalppo = 1000.0 
        returns_history_ppo = [] 
        rewards_history_ppo = []

        for t in range(window_size, data_length - 525, rebalance_period):
            date1 = t - rebalance_period

            s_ = env.get_state(t, window_size)
            action = agent.act(s_)

            ( actionppo, value, log_prob, raw_action ) = ppo_agent.act( s_, deterministic=False )

            if t == window_size:
                actions_to_show1 = np.zeros(N_ASSETS - 311)
            else:
                actions_to_show1 = actions_to_show[-1]

            actions_to_show.append(action[0])
            diffc.append(action[0] - actions_to_show1)

            ( weighted_returnsppo, rewardppo, WW0ppo, initial_capitalppo1 ) = env.get_reward( actionppo, date1, t, W0ppo, initial_capitalppo )

            weighted_returns, reward, WW0, initial_capital1 = env.get_reward(action[0], date1, t, W0, initial_capital)
            weighted_returns_equal, reward_equal, WW0eq, initial_capital1eq = env.get_reward(
                np.ones(agent.portfolio_size - 311) / (agent.portfolio_size - 311), date1, t, W0eq, initial_capitaleq)

            rewardppo = float( np.mean( np.asarray(rewardppo) ) )
            ppo_agent.remember( s_, raw_action, rewardppo, value, log_prob, False )

            returns_history_ppo.extend( weighted_returnsppo ) 
            rewards_history_ppo.append( rewardppo )

            W0ppo = WW0ppo 
            initial_capitalppo = ( initial_capitalppo1 )

            lookback = 63
            returns_window = env.get_state(t, window_size)
            returns_window = returns_window.iloc[ :, :N_ASSETS - 311 ]
            minvar_action = minimum_variance_weights(returns_window)

            weighted_returns_minvar, reward_minvar, WW0minvar, initial_capital_minvar =  env.get_reward( minvar_action, date1, t, W0minvar, initial_capital_minvar )

            rewards_history.append(reward)
            equal_rewards.append(reward_equal)
            minvar_rewards.append( reward_minvar )

            returns_history.extend(weighted_returns)
            returns_history_equal.extend(weighted_returns_equal)
            returns_history_minvar.extend( weighted_returns_minvar )

            W0 = WW0
            initial_capital = initial_capital1
            W0eq = WW0eq
            initial_capitaleq = initial_capital1eq
            W0minvar = WW0minvar
            initial_capital_minvar = initial_capital_minvar

            done = (t == data_length)

            agent.memory4replay.append((s, s_, action, reward, done))
            if len(agent.memory4replay) >= batch_size:
                agent.expReplay(batch_size)
                agent.memory4replay = []

            s = s_

        ppo_agent.memory[-1] = ( ppo_agent.memory[-1][0], ppo_agent.memory[-1][1], ppo_agent.memory[-1][2], ppo_agent.memory[-1][3], ppo_agent.memory[-1][4], True ) 
        ppo_agent.train( epochs=50 )
        print( "Episode finished." )

        # Plot results (optional, you can move outside the loop if desired)
        rl_result = np.array(returns_history).cumsum()
        equal_result = np.array(returns_history_equal).cumsum()
        minvar_result = ( np.array( returns_history_minvar ).cumsum() )

        plt.figure(figsize=(12, 2))
        plt.plot(rl_result, color='black', ls='-')
        plt.plot(equal_result, color='grey', ls='--')
        plt.title(f"Episode {e + 1} Cumulative Return")
        plt.show()

        plt.figure(figsize=(12, 2))
        for a in actions_to_show:
            plt.bar(np.arange(N_ASSETS - 311), a, color='grey', alpha=0.25)
            close_columns = [
                col for col in env.data.columns
                if col.startswith('<CLOSE>_') and col != '<CLOSE>_Overall Index'
            ]

            plt.xticks(
                np.arange(N_ASSETS - 311),
                env.data[close_columns].columns,
                rotation='vertical'
            )
        plt.title(f"Episode {e + 1} Portfolio Allocations")
        plt.show()

        """# 🎯 Compute average reward for this episode
        avg_reward = (np.mean(rewards_history)+np.mean(rewards_history_ppo))/2
        reward_list.append(avg_reward)

        # 💾 Save the model if it's the best so far
        if avg_reward > early_stop.best_score:
            #agent.model.save("best_agent_model.h5")
            agent.model.save_weights("best_weights.weights.h5")
            ppo_agent.save("best_ppo")
            if early_stop.verbose:
                print(f"📦 Model saved at episode {e + 1} (avg_reward: {avg_reward:.4f})")

        # 💡 Early Stopping Check
        if early_stop.check(avg_reward, e):
            print("✅ Early stopping triggered — training halted.")
            break"""

        # ============================================================
        # Separate reward for DQN and PPO
        # ============================================================

        avg_reward_dqn = np.mean(rewards_history)
        avg_reward_ppo = np.mean(rewards_history_ppo)

        reward_list_dqn.append(avg_reward_dqn)
        reward_list_ppo.append(avg_reward_ppo)

        print(
            f"Episode {e + 1}: "
            f"DQN reward = {avg_reward_dqn:.6f}, "
            f"PPO reward = {avg_reward_ppo:.6f}"
        )


        # ============================================================
        # DQN checkpoint + early stopping
        # ============================================================

        if avg_reward_dqn > early_stop_dqn.best_score:

            agent.model.save_weights("best_weights.weights.h5")

            if early_stop_dqn.verbose:
                print(
                    f"📦 Best DQN saved at episode {e + 1} "
                    f"(reward = {avg_reward_dqn:.6f})"
                )

        dqn_stop = early_stop_dqn.check(avg_reward_dqn, e)


        # ============================================================
        # PPO checkpoint + early stopping
        # ============================================================

        if avg_reward_ppo > early_stop_ppo.best_score:

            ppo_agent.save("best_ppo")

            if early_stop_ppo.verbose:
                print(
                    f"📦 Best PPO saved at episode {e + 1} "
                    f"(reward = {avg_reward_ppo:.6f})"
                )

        ppo_stop = early_stop_ppo.check(avg_reward_ppo, e)


        # ============================================================
        # Stop only when BOTH algorithms have converged
        # ============================================================

        if dqn_stop and ppo_stop:

            print("✅ DQN and PPO early stopping criteria both triggered.")
            print("Training halted.")

            break




    from tensorflow.keras.models import load_model

    # Load trained model

    """best_model = load_model("best_agent_model.h5")
    # Assign it to your agent before testing
    agent.model = best_model
    agent.is_eval = True
    agent.epsilon = 0  # Ensure greedy policy (no exploration)"""

    agent.model = agent._model()  # rebuild model architecture
    agent.model.load_weights("best_weights.weights.h5")
    agent.is_eval = True
    agent.epsilon = 0



    meanresult_equal, meanresult_rl, meanresult_minvar,meanresult_markowitz, meanresult_risk_parity,meanresult_max_sharpe,meanresult_rlppo=[],[], [],[],[],[],[]
    actions_equal, actions_rl,actions_minvar,actions_markowitz,actions_risk_parity,actions_max_sharpe,actions_rlppo  = [], [], [] ,[],[],[],[]
    result_equal, result_rl,result_minvar, result_markowitz,result_risk_parity, result_max_sharpe,result_rlppo= [], [],  [],[],[],[],[]
    cccc=[]
    actions_to_showtest=[]

    W0=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0 = [element * 0 for element in W0]
    W0eq=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0eq = [element * 0 for element in W0eq]
    W0minvar=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0minvar = [element * 0 for element in W0minvar]
    W0markowitz=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0markowitz = [element * 0 for element in W0markowitz]
    W0risk_parity=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0risk_parity = [element * 0 for element in W0risk_parity]
    W0max_sharpe=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0max_sharpe = [element * 0 for element in W0max_sharpe]
    W0ppo=(N_ASSETS-311)*[ 1./(N_ASSETS-311)]
    W0ppo = [element * 0 for element in W0ppo]

    initial_capitalppo=float(1000)
    initial_capital=float(1000)
    initial_capitaleq=float(1000)
    initial_capital_minvar = float(1000)
    initial_capital_markowitz = float(1000)
    initial_capital_risk_parity = float(1000)
    initial_capital_max_sharpe = float(1000)
    lookback = 63

    test_start = len(env.data) - 525
    test_end = len(env.data)
    t = test_start
    
    #for t in range(len(env.data)-505, len(env.data), rebalance_period):
    while t < test_end:


        #date1 = t-rebalance_period

        next_t = min(t + rebalance_period, test_end)
        date1 = t
        t = next_t


        s_ = env.get_state(t, window_size)
        action = agent.act(s_)
        actions_to_showtest.append(action[0])
        actionppo, value, log_prob, raw_action = ppo_agent.act(s_,deterministic=True)

        equal_action = np.ones(N_ASSETS - 311) / (N_ASSETS - 311)

        print("\n========== PPO DEBUG ==========")
        print("PPO action:")
        print(actionppo)

        print("\nEqual action:")
        print(equal_action)

        print("\nPPO sum:", np.sum(actionppo))
        print("PPO std:", np.std(actionppo))
        print("Equal std:", np.std(equal_action))

        print("Max abs difference:",np.max(np.abs(actionppo - equal_action)))

        print("Raw action:")
        print(raw_action)

        print("Raw action std:",np.std(raw_action))


        weighted_returnsppo, rewardppo, WW0ppo, initial_capitalppo1 = env.get_reward(actionppo,date1,t,W0ppo,initial_capitalppo)

        weighted_returns, reward,WW0,initial_capital1 = env.get_reward(action[0], date1, t,W0,initial_capital)
        weighted_returns_equal, reward_equal,WW0eq,initial_capital1eq = env.get_reward(
            np.ones((agent.portfolio_size)-311) / ((agent.portfolio_size)-311), date1, t,W0eq,initial_capitaleq)

        start_idx = max( 0, t - lookback )
        returns_window = env.get_state(t, window_size)
        returns_window = returns_window.iloc[ :, :((agent.portfolio_size)-311) ]

        minvar_action = minimum_variance_weights( returns_window )
        markowitz_action = markowitz_weights( returns_window )
        risk_parity_action = risk_parity_weights( returns_window )
        max_sharpe_action = max_sharpe_weights( returns_window )

        weighted_returns_minvar, reward_minvar,WW0minvar, initial_capital_minvar1 = env.get_reward( minvar_action, date1, t, W0minvar, initial_capital_minvar )
        weighted_returns_markowitz, reward_markowitz,WW0markowitz, initial_capital_markowitz1 = env.get_reward( markowitz_action, date1, t, W0markowitz, initial_capital_markowitz )
        weighted_returns_risk_parity, reward_risk_parity,WW0risk_parity, initial_capital_risk_parity1 = env.get_reward( risk_parity_action, date1, t, W0risk_parity, initial_capital_risk_parity )
        weighted_returns_max_sharpe, reward_max_sharpe,WW0max_sharpe, initial_capital_max_sharpe1 = env.get_reward( max_sharpe_action, date1, t, W0max_sharpe, initial_capital_max_sharpe )


        W0ppo=WW0ppo
        initial_capitalppo=initial_capitalppo1

        W0=WW0
        initial_capital=initial_capital1
        W0eq=WW0eq
        initial_capitaleq=initial_capital1eq

        # Minimum Variance
        W0minvar = WW0minvar
        initial_capital_minvar = initial_capital_minvar1
        # markowitz
        W0markowitz = WW0markowitz
        initial_capital_markowitz = initial_capital_markowitz1

        # risk_parity
        W0risk_parity = WW0risk_parity
        initial_capital_risk_parity = initial_capital_risk_parity1

        # max_sharpe
        W0max_sharpe = WW0max_sharpe
        initial_capital_max_sharpe = initial_capital_max_sharpe1

        result_equal.append(weighted_returns_equal.tolist())
        meanresult_equal.append(np.mean(np.array(weighted_returns_equal)))
        actions_equal.append(np.ones((agent.portfolio_size)-311) / ((agent.portfolio_size)-311))

        cccc.append(date1)

        result_rlppo.append(weighted_returnsppo.tolist())
        meanresult_rlppo.append(np.mean(np.array(weighted_returnsppo)))
        actions_rlppo.append(actionppo)

        result_rl.append(weighted_returns.tolist())
        meanresult_rl.append(np.mean(np.array(weighted_returns)))
        actions_rl.append(action[0])

        result_minvar.append( weighted_returns_minvar.tolist() )
        meanresult_minvar.append( np.mean( np.asarray( weighted_returns_minvar ) ) ) 
        actions_minvar.append( minvar_action )

        result_markowitz.append( weighted_returns_markowitz.tolist() )
        meanresult_markowitz.append( np.mean( np.asarray( weighted_returns_markowitz ) ) ) 
        actions_markowitz.append( markowitz_action )

        result_risk_parity.append( weighted_returns_risk_parity.tolist() )
        meanresult_risk_parity.append( np.mean( np.asarray( weighted_returns_risk_parity ) ) ) 
        actions_risk_parity.append( risk_parity_action )

        result_max_sharpe.append( weighted_returns_max_sharpe.tolist() )
        meanresult_max_sharpe.append( np.mean( np.asarray( weighted_returns_max_sharpe ) ) ) 
        actions_max_sharpe.append( max_sharpe_action )

    # ==============================
    # Market Index Returns
    # ==============================

    dataind = read_csv(
        'indexnew.csv',
        index_col=0,
        header=2
    ).ffill().bfill()

    dataind = dataind.rename(
        columns={'Unnamed: 1': 'Overall Index'}
    )

    dataind = dataind[len(env.data)-525:len(env.data)]

    dataind = dataind.reset_index(drop=True)

    # Daily market returns
    xxx = (
        dataind['Overall Index']
        .pct_change()
        .fillna(0).replace([np.inf, -np.inf], 0)
        .to_numpy()
    )






    result_equal_vis = [item for sublist in result_equal for item in sublist]
    result_rl_vis = [item for sublist in result_rl for item in sublist]
    result_minvar_vis = [item for sublist in result_minvar for item in sublist]
    result_markowitz_vis = [item for sublist in result_markowitz for item in sublist]
    result_risk_parity_vis = [item for sublist in result_risk_parity for item in sublist]
    result_max_sharpe_vis = [item for sublist in result_max_sharpe for item in sublist]
    result_rlppo_vis = [item for sublist in result_rlppo for item in sublist]

    print("len(result_equal_vis) =", len(result_equal_vis))
    print("len(result_minvar_vis) =", len(result_minvar_vis))
    print("len(result_markowitz_vis) =", len(result_markowitz_vis))
    print("len(result_risk_parity_vis) =", len(result_risk_parity_vis))
    print("len(result_max_sharpe_vis) =", len(result_max_sharpe_vis))

    print("len(xxx) =", len(xxx))

    # Importing library
    import scipy.stats as stats

    # pre holds the mileage before
    # applying the different engine oil
    pre = result_equal_vis

    # post holds the mileage after
    # applying the different engine oil
    post = result_rl_vis
    # Performing the paired sample t-test
    stats.ttest_rel(pre, post,alternative='less')



    plt.figure(figsize = (10, 5))
    plt.plot(np.array(result_equal_vis).cumsum(), label = 'Benchmark', color = 'grey',ls = '--')
    plt.plot(np.array(result_rl_vis).cumsum(), label = 'Deep RL portfolio', color = 'black',ls = '-')
    plt.plot(np.array(xxx).cumsum(), label = 'Index', color = 'green',ls = '-.')
    df11133445566=pd.DataFrame(np.array(xxx).cumsum(),columns=['Index']).join(pd.DataFrame(np.array(result_equal_vis).cumsum(),columns=['Benchmark']))
    df1113344556677=pd.DataFrame(np.array(result_rl_vis).cumsum(),columns=['Deep RL portfolio']).join(df11133445566)
    for var in list(['Benchmark','Deep RL portfolio','Index']):
        plt.annotate('%0.2f' % df1113344556677[var].iloc[-1], xy=(1, df1113344556677[var].iloc[-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points',size=10, va="center")
    plt.xlabel('Date')
    # Set the y axis label of the current axis.
    plt.ylabel('Cumluative_Return')
    # Set a title of the current axes.
    plt.title('Comparison of Performance')
    # show a legend on the plot
    plt.legend()
    plt.show()



    #Plotting the data
    """import matplotlib
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='red')

    N = len(np.array([item for sublist in result_equal for item in sublist]).cumsum())

    for i in range(0, len(actions_rl)):
        current_range = np.arange(0, N)
        current_ts = np.zeros(N)
        current_ts2 = np.zeros(N)

        ts_benchmark = np.array([item for sublist in result_equal[:i+1] for item in sublist]).cumsum()
        ts_target = np.array([item for sublist in result_rl[:i+1] for item in sublist]).cumsum()

        t = len(ts_benchmark)
        current_ts[:t] = ts_benchmark
        current_ts2[:t] = ts_target

        current_ts[current_ts == 0] = ts_benchmark[-1]
        current_ts2[current_ts2 == 0] = ts_target[-1]

        plt.figure(figsize = (12, 10))

        plt.subplot(2, 1, 1)
        plt.bar(np.arange(N_ASSETS-191), actions_rl[i], color = 'grey')
        plt.xticks(np.arange(N_ASSETS-191), env.data[['<CLOSE>_BANK','<CLOSE>_BMLT','<CLOSE>_DTIP',
                                                        '<CLOSE>_FOLD','<CLOSE>_GDIR','<CLOSE>_MADN',
                                                        '<CLOSE>_MAPN','<CLOSE>_MSMI','<CLOSE>_NORI',
                                                        '<CLOSE>_OIMC','<CLOSE>_PASN','<CLOSE>_PKLJ',
                                                        '<CLOSE>_PNES','<CLOSE>_PTAP','<CLOSE>_SFKZ',
                                                        '<CLOSE>_SSAP','<CLOSE>_TAMN','<CLOSE>_TSBE']].columns, rotation='vertical')
        plt.xlabel('Stock')
        # Set the y axis label of the current axis.
        plt.ylabel('RL_Action')
        # Set a title of the current axes.
        plt.title('Comparison of Performance')

        plt.subplot(2, 1, 2)
        plt.colormaps = current_cmap
        plt.plot((current_range[:t]), current_ts[:t], color = 'black', label = 'Benchmark')
        plt.plot((current_range[:t]), current_ts2[:t], color = 'red', label = 'Deep RL portfolio')
        plt.plot((current_range[t:]), current_ts[t:], ls = '--', lw = .1, color = 'black')
        #plt.autoscale(False)
        plt.ylim([-0.8, 0.8])
        plt.xlabel('Date')
        # Set the y axis label of the current axis.
        plt.ylabel('Cumluative_Return')
        # Set a title of the current axes.
        #plt.title('Comparison of Performance')
        # show a legend on the plot
        plt.legend()"""



    import statsmodels.api as sm
    from statsmodels import regression
    rf=0.23
    def sharpe(R):
        #r = np.diff(R)
        sr = (R.mean()*252-rf)/(R.std() * np.sqrt(252))
        return sr

    def sortino_ratio(series):
        #series = np.diff(series)
        mean = series.mean() * 252
        std_neg = series[series<0].std() * np.sqrt(252)
        sor=(mean-rf)/std_neg
        return sor

    def max_drawdown(return_series):
        comp_ret = pd.Series((return_series+1).cumprod())
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret/peak)-1
        return dd.min()

    def print_stats(result, benchmark):

        sharpe_ratio = sharpe(np.array(result))
        sor = sortino_ratio(np.array(result))
        max_draw=max_drawdown(np.array(result))
        returns = np.mean(np.array(result)*252)
        volatility = np.std(np.array(result)*np.sqrt(252))
        calmars = (returns-rf)/abs(max_draw)

        X = benchmark
        y = result
        x = sm.add_constant(X)
        """print("======================================")
        print("DEBUG OLS")
        print("len(y) =", len(y))
        print("len(x) =", len(x))
        print("shape(y) =", np.asarray(y).shape)
        print("shape(x) =", np.asarray(x).shape)
        print("======================================")"""
        model = regression.linear_model.OLS(y, x).fit()
        alpha = model.params[0]*252
        beta = model.params[1]



        return np.round(np.array([returns, volatility, sharpe_ratio,sor,max_draw,calmars, alpha, beta]), 4).tolist()


    """print('EQUAL', print_stats(result_equal_vis, xxx))
    print('RL AGENT', print_stats(result_rl_vis, xxx))
    print('MINIMUM VARIANCE', print_stats(result_minvar_vis, xxx))
    print('MARKOWITZ', print_stats(result_markowitz_vis, xxx))
    print('RISK_PARITY', print_stats(result_risk_parity_vis, xxx))
    print('MAX_SHARPE', print_stats(result_max_sharpe_vis, xxx))
    print('RLPPO AGENT', print_stats(result_rlppo_vis, xxx))
    print('index', print_stats(xxx, xxx))"""


    filename = f'results_cnntwo{ttt}.txt'

    with open(filename, 'w', encoding='utf-8') as f:

        f.write('EQUAL\n')
        f.write(str(print_stats(result_equal_vis, xxx)))
        f.write('\n\n')

        f.write('RL AGENT\n')
        f.write(str(print_stats(result_rl_vis, xxx)))
        f.write('\n\n')

        f.write('MINIMUM VARIANCE\n')
        f.write(str(print_stats(result_minvar_vis, xxx)))
        f.write('\n\n')

        f.write('MARKOWITZ\n')
        f.write(str(print_stats(result_markowitz_vis, xxx)))
        f.write('\n\n')

        f.write('RISK_PARITY\n')
        f.write(str(print_stats(result_risk_parity_vis, xxx)))
        f.write('\n\n')

        f.write('MAX_SHARPE\n')
        f.write(str(print_stats(result_max_sharpe_vis, xxx)))
        f.write('\n\n')

        f.write('RLPPO AGENT\n')
        f.write(str(print_stats(result_rlppo_vis, xxx)))
        f.write('\n\n')

        f.write('INDEX\n')
        f.write(str(print_stats(xxx, xxx)))
        f.write('\n\n')

    print(f'Saved: {filename}')


    pd.DataFrame(result_rl_vis).to_csv(
        f'result_rl_cnntwo{ttt}.csv'
    )

    pd.DataFrame(result_rlppo_vis).to_csv(
        f'result_rlppo_cnntwo{ttt}.csv'
    )

    pd.DataFrame(result_minvar_vis).to_csv(
        f'result_minvartwo{ttt}.csv',
        index=False
    )

    pd.DataFrame(result_markowitz_vis).to_csv(
        f'result_markowitztwo{ttt}.csv',
        index=False
    )

    pd.DataFrame(result_risk_parity_vis).to_csv(
        f'result_risk_paritytwo{ttt}.csv',
        index=False
    )

    pd.DataFrame(result_max_sharpe_vis).to_csv(
        f'result_max_sharpetwo{ttt}.csv',
        index=False
    )


    pd.DataFrame(result_equal_vis).to_csv('result_equaltwo.csv')
    pd.DataFrame(xxx).to_csv('xxxtwo.csv')


"""
rl_cnn1= read_csv('result_rl_cnn1.csv',index_col=0)
rl_cnn1.columns = ['result_rl_cnn1']
rl_cnn2= read_csv('result_rl_cnn2.csv',index_col=0)
rl_cnn2.columns = ['result_rl_cnn2']
rl_cnn3= read_csv('result_rl_cnn3.csv',index_col=0)
rl_cnn3.columns = ['result_rl_cnn3']
rl_cnn4= read_csv('result_rl_cnn4.csv',index_col=0)
rl_cnn4.columns = ['result_rl_cnn4']
rl_cnn5= read_csv('result_rl_cnn5.csv',index_col=0)
rl_cnn5.columns = ['result_rl_cnn5']
rl_cnn6= read_csv('result_rl_cnn6.csv',index_col=0)
rl_cnn6.columns = ['result_rl_cnn6']
rl_cnn7= read_csv('result_rl_cnn7.csv',index_col=0)
rl_cnn7.columns = ['result_rl_cnn7']


rl_cnnlstm1= read_csv('result_rl_cnnlstm1.csv',index_col=0)
rl_cnnlstm1.columns = ['result_rl_cnnlstm1']
rl_cnnlstm2= read_csv('result_rl_cnnlstm2.csv',index_col=0)
rl_cnnlstm2.columns = ['result_rl_cnnlstm2']
rl_cnnlstm3= read_csv('result_rl_cnnlstm3.csv',index_col=0)
rl_cnnlstm3.columns = ['result_rl_cnnlstm3']
rl_cnnlstm4= read_csv('result_rl_cnnlstm4.csv',index_col=0)
rl_cnnlstm4.columns = ['result_rl_cnnlstm4']
rl_cnnlstm5= read_csv('result_rl_cnnlstm5.csv',index_col=0)
rl_cnnlstm5.columns = ['result_rl_cnnlstm5']
rl_cnnlstm6= read_csv('result_rl_cnnlstm6.csv',index_col=0)
rl_cnnlstm6.columns = ['result_rl_cnnlstm6']
rl_cnnlstm7= read_csv('result_rl_cnnlstm7.csv',index_col=0)
rl_cnnlstm7.columns = ['result_rl_cnnlstm7']


rl_cnntransformer1= read_csv('result_rl_cnntransformer1.csv',index_col=0)
rl_cnntransformer1.columns = ['result_rl_cnntransformer1']
rl_cnntransformer2= read_csv('result_rl_cnntransformer2.csv',index_col=0)
rl_cnntransformer2.columns = ['result_rl_cnntransformer2']
rl_cnntransformer3= read_csv('result_rl_cnntransformer3.csv',index_col=0)
rl_cnntransformer3.columns = ['result_rl_cnntransformer3']
rl_cnntransformer4= read_csv('result_rl_cnntransformer4.csv',index_col=0)
rl_cnntransformer4.columns = ['result_rl_cnntransformer4']
rl_cnntransformer5= read_csv('result_rl_cnntransformer5.csv',index_col=0)
rl_cnntransformer5.columns = ['result_rl_cnntransformer5']
rl_cnntransformer6= read_csv('result_rl_cnntransformer6.csv',index_col=0)
rl_cnntransformer6.columns = ['result_rl_cnntransformer6']
rl_cnntransformer7= read_csv('result_rl_cnntransformer7.csv',index_col=0)
rl_cnntransformer7.columns = ['result_rl_cnntransformer7']



rl_cnn1= read_csv('result_rl_cnntwe1.csv',index_col=0)
rl_cnn1.columns = ['result_rl_cnntwe1']
rl_cnn2= read_csv('result_rl_cnntwe2.csv',index_col=0)
rl_cnn2.columns = ['result_rl_cnntwe2']
rl_cnn3= read_csv('result_rl_cnntwe3.csv',index_col=0)
rl_cnn3.columns = ['result_rl_cnntwe3']
rl_cnn4= read_csv('result_rl_cnntwe4.csv',index_col=0)
rl_cnn4.columns = ['result_rl_cnntwe4']
rl_cnn5= read_csv('result_rl_cnntwe5.csv',index_col=0)
rl_cnn5.columns = ['result_rl_cnntwe5']
rl_cnn6= read_csv('result_rl_cnntwe6.csv',index_col=0)
rl_cnn6.columns = ['result_rl_cnntwe6']
rl_cnn7= read_csv('result_rl_cnntwo7.csv',index_col=0)
rl_cnn7.columns = ['result_rl_cnntwo7']


rl_cnnlstm1= read_csv('result_rl_cnnlstmtwo1.csv',index_col=0)
rl_cnnlstm1.columns = ['result_rl_cnnlstmtwo1']
rl_cnnlstm2= read_csv('result_rl_cnnlstmtwo2.csv',index_col=0)
rl_cnnlstm2.columns = ['result_rl_cnnlstmtwo2']
rl_cnnlstm3= read_csv('result_rl_cnnlstmtwo3.csv',index_col=0)
rl_cnnlstm3.columns = ['result_rl_cnnlstmtwo3']
rl_cnnlstm4= read_csv('result_rl_cnnlstmtwo4.csv',index_col=0)
rl_cnnlstm4.columns = ['result_rl_cnnlstmtwo4']
rl_cnnlstm5= read_csv('result_rl_cnnlstmtwo5.csv',index_col=0)
rl_cnnlstm5.columns = ['result_rl_cnnlstmtwo5']
rl_cnnlstm6= read_csv('result_rl_cnnlstmtwo6.csv',index_col=0)
rl_cnnlstm6.columns = ['result_rl_cnnlstmtwo6']
rl_cnnlstm7= read_csv('result_rl_cnnlstmtwo7.csv',index_col=0)
rl_cnnlstm7.columns = ['result_rl_cnnlstmtwo7']


rl_cnntransformer1= read_csv('result_rl_cnntransformertwo1.csv',index_col=0)
rl_cnntransformer1.columns = ['result_rl_cnntransformertwo1']
rl_cnntransformer2= read_csv('result_rl_cnntransformertwo2.csv',index_col=0)
rl_cnntransformer2.columns = ['result_rl_cnntransformertwo2']
rl_cnntransformer3= read_csv('result_rl_cnntransformertwo3.csv',index_col=0)
rl_cnntransformer3.columns = ['result_rl_cnntransformertwo3']
rl_cnntransformer4= read_csv('result_rl_cnntransformertwo4.csv',index_col=0)
rl_cnntransformer4.columns = ['result_rl_cnntransformertwo4']
rl_cnntransformer5= read_csv('result_rl_cnntransformertwo5.csv',index_col=0)
rl_cnntransformer5.columns = ['result_rl_cnntransformertwo5']
rl_cnntransformer6= read_csv('result_rl_cnntransformertwo6.csv',index_col=0)
rl_cnntransformer6.columns = ['result_rl_cnntransformertwo6']
rl_cnntransformer7= read_csv('result_rl_cnntransformertwo7.csv',index_col=0)
rl_cnntransformer7.columns = ['result_rl_cnntransformertwo7']

rl_equaltwo= read_csv('result_equaltwo.csv',index_col=0)
rl_equaltwo.columns = ['result_equaltwo']

vnp=pd.concat([rl_cnn1.cumsum(),rl_cnn2.cumsum(),rl_cnn3.cumsum(),rl_cnn4.cumsum(),rl_cnn5.cumsum(),rl_cnn6.cumsum(),rl_cnn7.cumsum(),],axis=1)
plt.figure(figsize = (10, 5))
plt.plot(rl_cnn1.cumsum(), label = 'rl_1',ls = '--')
plt.plot(rl_cnn2.cumsum(), label = 'rl_2',ls = '-')
plt.plot(rl_cnn3.cumsum(), label = 'rl_3',ls = '-.')
plt.plot(rl_cnn4.cumsum(), label = 'rl_4',ls = 'dashdot')
plt.plot(rl_cnn5.cumsum(), label = 'rl_5',ls = ':')
plt.plot(rl_cnn6.cumsum(), label = 'rl_6',ls = 'solid')
plt.plot(rl_cnn7.cumsum(), label = 'rl_7',ls = 'dashed')
for var in list(vnp.columns):
    plt.annotate('%0.2f' % vnp[var].iloc[-1], xy=(1, vnp[var].iloc[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points',size=10, va="center")
plt.xlabel('Date')
# Set the y axis label of the current axis.
plt.ylabel('Cumluative_Return')
# Set a title of the current axes.
plt.title('Comparison of Performance')
# show a legend on the plot
plt.legend()
plt.show()



vnp=pd.concat([rl_cnnlstm1.cumsum(),rl_cnnlstm2.cumsum(),rl_cnnlstm3.cumsum(),rl_cnnlstm4.cumsum(),rl_cnnlstm5.cumsum(),rl_cnnlstm6.cumsum(),rl_cnnlstm7.cumsum(),],axis=1)
plt.figure(figsize = (10, 5))
plt.plot(rl_cnnlstm1.cumsum(), label = 'rl_1',ls = '--')
plt.plot(rl_cnnlstm2.cumsum(), label = 'rl_2',ls = '-')
plt.plot(rl_cnnlstm3.cumsum(), label = 'rl_3',ls = '-.')
plt.plot(rl_cnnlstm4.cumsum(), label = 'rl_4',ls = 'dashdot')
plt.plot(rl_cnnlstm5.cumsum(), label = 'rl_5',ls = ':')
plt.plot(rl_cnnlstm6.cumsum(), label = 'rl_6',ls = 'solid')
plt.plot(rl_cnnlstm7.cumsum(), label = 'rl_7',ls = 'dashed')
for var in list(vnp.columns):
    plt.annotate('%0.2f' % vnp[var].iloc[-1], xy=(1, vnp[var].iloc[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points',size=10, va="center")
plt.xlabel('Date')
# Set the y axis label of the current axis.
plt.ylabel('Cumluative_Return')
# Set a title of the current axes.
plt.title('Comparison of Performance')
# show a legend on the plot
plt.legend()
plt.show()



vnp=pd.concat([rl_cnntransformer1.cumsum(),rl_cnntransformer2.cumsum(),rl_cnntransformer3.cumsum(),rl_cnntransformer4.cumsum(),rl_cnntransformer5.cumsum(),rl_cnntransformer6.cumsum(),rl_cnntransformer7.cumsum(),],axis=1)
plt.figure(figsize = (10, 5))
plt.plot(rl_cnntransformer1.cumsum(), label = 'rl_1',ls = '--')
plt.plot(rl_cnntransformer2.cumsum(), label = 'rl_2',ls = '-')
plt.plot(rl_cnntransformer3.cumsum(), label = 'rl_3',ls = '-.')
plt.plot(rl_cnntransformer4.cumsum(), label = 'rl_4',ls = 'dashdot')
plt.plot(rl_cnntransformer5.cumsum(), label = 'rl_5',ls = ':')
plt.plot(rl_cnntransformer6.cumsum(), label = 'rl_6',ls = 'solid')
plt.plot(rl_cnntransformer7.cumsum(), label = 'rl_7',ls = 'dashed')
for var in list(vnp.columns):
    plt.annotate('%0.2f' % vnp[var].iloc[-1], xy=(1, vnp[var].iloc[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points',size=10, va="center")
plt.xlabel('Date')
# Set the y axis label of the current axis.
plt.ylabel('Cumluative_Return')
# Set a title of the current axes.
plt.title('Comparison of Performance')
# show a legend on the plot
plt.legend()
plt.show()



# Importing library
import scipy.stats as stats

# pre holds the mileage before
# applying the different engine oil
pre = rl_cnntransformer1
# post holds the mileage after
# applying the different engine oil
post = rl_equaltwo
# Performing the paired sample t-test
stats.ttest_rel(pre, post,alternative='less')



# Importing library
import scipy.stats as stats

# pre holds the mileage before
# applying the different engine oil
pre = rl_cnntransformer3
# post holds the mileage after
# applying the different engine oil
post = rl_cnntransformer1
# Performing the paired sample t-test
stats.ttest_rel(pre, post,alternative='greater')"""

