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



rl_cnn1= read_csv('result_rl_cnntwo0.csv',index_col=0)
rl_cnn1.columns = ['result_rl_cnntwo0']
rl_cnn2= read_csv('result_rl_cnntwo1.csv',index_col=0)
rl_cnn2.columns = ['result_rl_cnntwo1']
rl_cnn3= read_csv('result_rl_cnntwo2.csv',index_col=0)
rl_cnn3.columns = ['result_rl_cnntwo2']
rl_cnn4= read_csv('result_rl_cnntwo3.csv',index_col=0)
rl_cnn4.columns = ['result_rl_cnntwo3']
rl_cnn5= read_csv('result_rl_cnntwo4.csv',index_col=0)
rl_cnn5.columns = ['result_rl_cnntwo4']
rl_cnn6= read_csv('result_rl_cnntwo5.csv',index_col=0)
rl_cnn6.columns = ['result_rl_cnntwo5']
rl_cnn7= read_csv('result_rl_cnntwo6.csv',index_col=0)
rl_cnn7.columns = ['result_rl_cnntwo6']
rl_cnn8= read_csv('result_rl_cnntwo7.csv',index_col=0)
rl_cnn8.columns = ['result_rl_cnntwo7']
rl_cnn9= read_csv('result_rl_cnntwo8.csv',index_col=0)
rl_cnn9.columns = ['result_rl_cnntwo8']
rl_cnn10= read_csv('result_rl_cnntwo9.csv',index_col=0)
rl_cnn10.columns = ['result_rl_cnntwo9']
rl_cnn11= read_csv('result_rl_cnntwo10.csv',index_col=0)
rl_cnn11.columns = ['result_rl_cnntwo10']
rl_cnn12= read_csv('result_rl_cnntwo11.csv',index_col=0)
rl_cnn12.columns = ['result_rl_cnntwo11']
rl_cnn13= read_csv('result_rl_cnntwo12.csv',index_col=0)
rl_cnn13.columns = ['result_rl_cnntwo12']
rl_cnn14= read_csv('result_rl_cnntwo13.csv',index_col=0)
rl_cnn14.columns = ['result_rl_cnntwo13']

rl_equaltwo= read_csv('result_equaltwo.csv',index_col=0)
rl_equaltwo.columns = ['result_equaltwo']
rl_xxxtwo= read_csv('xxxtwo.csv',index_col=0)
rl_xxxtwo.columns = ['xxxtwo']

rl_markowitztwo= read_csv('result_markowitztwo0.csv',)
rl_markowitztwo.columns = ['result_markowitztwo0']

rl_max_sharpetwo= read_csv('result_max_sharpetwo0.csv',)
rl_max_sharpetwo.columns = ['result_max_sharpetwo0']

rl_minvartwo= read_csv('result_minvartwo0.csv',)
rl_minvartwo.columns = ['result_minvartwo0']

rl_risk_paritytwo= read_csv('result_risk_paritytwo0.csv',)
rl_risk_paritytwo.columns = ['result_risk_paritytwo0']




"""vnp=pd.concat([rl_cnn1.cumsum(),rl_cnn2.cumsum(),rl_cnn3.cumsum(),rl_cnn4.cumsum(),rl_cnn5.cumsum(),rl_cnn6.cumsum(),rl_cnn7.cumsum(),],axis=1)
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
plt.show()"""



import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv

# ==========================================
# تابع محاسبه بازده انباشته
# ==========================================

def cumulative_return(series):
    return (1 + series).cumprod() - 1


# ==========================================
# خواندن 14 نتیجه CNN
# ==========================================

rl_cnn = {}

for i in range(14):
    filename = f'result_rl_cnntwo{i}.csv'
    
    temp = read_csv(filename, index_col=0)
    temp.columns = [f'result_rl_cnntwo{i}']
    
    rl_cnn[i] = temp.iloc[:, 0]


# ==========================================
# خواندن Benchmark ها
# ==========================================

rl_equal = read_csv(
    'result_equaltwo.csv',
    index_col=0
).iloc[:, 0]

rl_index = read_csv(
    'xxxtwo.csv',
    index_col=0
).iloc[:, 0]

rl_markowitz = read_csv(
    'result_markowitztwo0.csv'
).iloc[:, 0]

rl_max_sharpe = read_csv(
    'result_max_sharpetwo0.csv'
).iloc[:, 0]

rl_minvar = read_csv(
    'result_minvartwo0.csv'
).iloc[:, 0]

rl_risk_parity = read_csv(
    'result_risk_paritytwo0.csv'
).iloc[:, 0]


# ==========================================
# تبدیل Daily Return به Cumulative Return
# ==========================================

for i in range(14):
    rl_cnn[i] = cumulative_return(rl_cnn[i])

rl_equal = cumulative_return(rl_equal)
rl_index = cumulative_return(rl_index)
rl_markowitz = cumulative_return(rl_markowitz)
rl_max_sharpe = cumulative_return(rl_max_sharpe)
rl_minvar = cumulative_return(rl_minvar)
rl_risk_parity = cumulative_return(rl_risk_parity)


# ==========================================
# رسم نمودار
# ==========================================

plt.figure(figsize=(16, 8))

# ------------------------------------------
# 14 مدل CNN
# ------------------------------------------

for i in range(14):
    plt.plot(
        rl_cnn[i].values,
        linewidth=1.2,
        label=f'CNN-{i}'
    )


# ------------------------------------------
# Benchmark ها
# ------------------------------------------

plt.plot(
    rl_equal.values,
    linewidth=2.5,
    linestyle='--',
    label='Equal Weight'
)

plt.plot(
    rl_index.values,
    linewidth=2.5,
    linestyle=':',
    label='Market Index'
)

plt.plot(
    rl_markowitz.values,
    linewidth=2.5,
    linestyle='--',
    label='Markowitz'
)

plt.plot(
    rl_max_sharpe.values,
    linewidth=2.5,
    linestyle='--',
    label='Max Sharpe'
)

plt.plot(
    rl_minvar.values,
    linewidth=2.5,
    linestyle='--',
    label='Minimum Variance'
)

plt.plot(
    rl_risk_parity.values,
    linewidth=2.5,
    linestyle='--',
    label='Risk Parity'
)


# ==========================================
# تنظیمات نمودار
# ==========================================

plt.xlabel('Trading Days', fontsize=12)

plt.ylabel('Cumulative Return', fontsize=12)

plt.title(
    'Cumulative Return of CNN-Based RL Models and Benchmark Portfolios',
    fontsize=14
)

plt.grid(True, alpha=0.3)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    fontsize=9
)

plt.tight_layout()


# ==========================================
# ذخیره نمودار روی سرور
# ==========================================

plt.savefig(
    'CNN_RL_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()




import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.05
N_REWARDS = 14


# ============================================================
# FILES
# ============================================================

# ------------------------------------------------------------
# DQN / CNN-RL
# ------------------------------------------------------------

dqn_files = {
    f'DQN-R{i}': (
        f'result_rl_cnntwo{i}.csv',
        '0'
    )
    for i in range(N_REWARDS)
}


# ------------------------------------------------------------
# PPO
# ------------------------------------------------------------

ppo_files = {
    f'PPO-R{i}': (
        f'result_rlppo_cnntwo{i}.csv',
        '0'
    )
    for i in range(N_REWARDS)
}


# ------------------------------------------------------------
# BENCHMARKS
# ------------------------------------------------------------

benchmark_files = {

    'Markowitz': (
        'result_markowitztwo0.csv',
        '0'
    ),

    'Minimum Variance': (
        'result_minvartwo0.csv',
        '0'
    ),

    'Risk Parity': (
        'result_risk_paritytwo0.csv',
        '0'
    ),

    'Max Sharpe': (
        'result_max_sharpetwo0.csv',
        '0'
    ),

    'Equal Weight': (
        'result_equaltwo.csv',
        '0'
    ),

    'Index': (
        'xxxtwo.csv',
        '0'
    )
}


# ============================================================
# P-VALUE FORMAT
# ============================================================

def format_pvalue(p):

    if p < 0.001:
        return "< 0.001"

    return f"{p:.4f}"


# ============================================================
# SAFE WILCOXON
# ============================================================

def safe_wilcoxon(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    differences = x - y

    # حذف اختلاف‌های صفر
    nonzero = differences[differences != 0]

    # اگر همه اختلاف‌ها صفر باشند
    if len(nonzero) == 0:

        return {
            'statistic': 0.0,
            'pvalue': 1.0,
            'n_total': len(differences),
            'n_nonzero': 0,
            'median_difference': 0.0
        }

    statistic, pvalue = wilcoxon(
        nonzero,
        alternative='two-sided'
    )

    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'n_total': len(differences),
        'n_nonzero': len(nonzero),
        'median_difference': np.median(differences)
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_files(file_dict):

    data_dict = {}

    for name, (filename, column) in file_dict.items():

        try:

            df = pd.read_csv(filename)

            if column not in df.columns:

                raise ValueError(
                    f"Column '{column}' not found in {filename}. "
                    f"Available columns: {list(df.columns)}"
                )

            series = pd.to_numeric(
                df[column],
                errors='coerce'
            )

            data_dict[name] = series

            print(
                f"Loaded: {name:<22} "
                f"{filename:<40} "
                f"N = {series.notna().sum()}"
            )

        except Exception as e:

            print(
                f"\nERROR loading {name}: {filename}"
            )

            print(e)

            raise

    return pd.DataFrame(data_dict)


# ============================================================
# LOAD DQN
# ============================================================

print("\n" + "=" * 80)
print("LOADING DQN / CNN-RL DATA")
print("=" * 80)

dqn_data = load_files(dqn_files)


# ============================================================
# LOAD PPO
# ============================================================

print("\n" + "=" * 80)
print("LOADING PPO DATA")
print("=" * 80)

ppo_data = load_files(ppo_files)


# ============================================================
# LOAD BENCHMARKS
# ============================================================

print("\n" + "=" * 80)
print("LOADING BENCHMARK DATA")
print("=" * 80)

benchmark_data = load_files(benchmark_files)


# ============================================================
# ALIGN DQN DATA
# ============================================================

dqn_data = dqn_data.dropna()

print("\n" + "=" * 80)
print("DQN DATA INFORMATION")
print("=" * 80)

print(
    "Number of common observations:",
    len(dqn_data)
)

print(
    "Number of reward functions:",
    len(dqn_data.columns)
)


# ============================================================
# ALIGN PPO DATA
# ============================================================

ppo_data = ppo_data.dropna()

print("\n" + "=" * 80)
print("PPO DATA INFORMATION")
print("=" * 80)

print(
    "Number of common observations:",
    len(ppo_data)
)

print(
    "Number of reward functions:",
    len(ppo_data.columns)
)


# ============================================================
# DQN AND PPO COMMON DATES/OBSERVATIONS
# ============================================================
#
# برای مقایسه DQN-Ri با PPO-Ri باید همان مشاهدات/روزها
# استفاده شوند.
#
# چون فایل‌ها index زمانی را در ستون جداگانه نگه می‌دارند
# و ستون '0' فقط بازده است، در اینجا بر اساس طول و ترتیب
# مشاهدات مشترک کار می‌کنیم.
#
# اگر تمام فایل‌ها دقیقاً مربوط به همان 525 روز باشند،
# این بخش 525 مشاهده مشترک خواهد داد.
# ============================================================

n_common_rl = min(
    len(dqn_data),
    len(ppo_data)
)

dqn_common = dqn_data.iloc[
    :n_common_rl
].reset_index(drop=True)

ppo_common = ppo_data.iloc[
    :n_common_rl
].reset_index(drop=True)


print("\n" + "=" * 80)
print("DQN-PPO COMMON OBSERVATIONS")
print("=" * 80)

print(
    "Common observations:",
    n_common_rl
)


# ============================================================
# DESCRIPTIVE STATISTICS - DQN
# ============================================================

print("\n" + "=" * 80)
print("DQN DESCRIPTIVE STATISTICS")
print("=" * 80)

dqn_descriptive = dqn_data.describe().T[
    [
        'count',
        'mean',
        'std',
        'min',
        '50%',
        'max'
    ]
]

print(
    dqn_descriptive.to_string()
)


dqn_descriptive.to_csv(
    'dqn_14_rewards_descriptive_statistics.csv'
)


# ============================================================
# DESCRIPTIVE STATISTICS - PPO
# ============================================================

print("\n" + "=" * 80)
print("PPO DESCRIPTIVE STATISTICS")
print("=" * 80)

ppo_descriptive = ppo_data.describe().T[
    [
        'count',
        'mean',
        'std',
        'min',
        '50%',
        'max'
    ]
]

print(
    ppo_descriptive.to_string()
)


ppo_descriptive.to_csv(
    'ppo_14_rewards_descriptive_statistics.csv'
)


# ============================================================
# FRIEDMAN ANALYSIS FUNCTION
# ============================================================

def run_friedman_analysis(
    data,
    model_name
):

    methods = list(data.columns)

    print("\n" + "=" * 80)
    print(f"{model_name}: FRIEDMAN TEST - 14 REWARD FUNCTIONS")
    print("=" * 80)

    print(
        "Number of reward functions:",
        len(methods)
    )

    print(
        "Number of observations:",
        len(data)
    )

    # --------------------------------------------------------
    # Friedman
    # --------------------------------------------------------

    friedman_stat, friedman_p = friedmanchisquare(
        *[
            data[m].values
            for m in methods
        ]
    )

    df_friedman = len(methods) - 1

    print(
        f"\nChi-square statistic: "
        f"{friedman_stat:.6f}"
    )

    print(
        f"Degrees of freedom: "
        f"{df_friedman}"
    )

    print(
        f"p-value: "
        f"{format_pvalue(friedman_p)}"
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if friedman_p < ALPHA:

        print(
            "\nResult:"
        )

        print(
            "There is a statistically significant "
            "overall difference among the 14 reward functions."
        )

        friedman_significant = True

    else:

        print(
            "\nResult:"
        )

        print(
            "There is no statistically significant "
            "overall difference among the 14 reward functions."
        )

        friedman_significant = False


    # --------------------------------------------------------
    # Save Friedman result
    # --------------------------------------------------------

    friedman_result = pd.DataFrame({

        'Model':
            [model_name],

        'Number of Rewards':
            [len(methods)],

        'N':
            [len(data)],

        'Friedman Chi-square':
            [friedman_stat],

        'df':
            [df_friedman],

        'p-value':
            [friedman_p],

        'Significant at 5%':
            [friedman_p < ALPHA]

    })

    friedman_result.to_csv(
        f'{model_name.lower()}_friedman_14_rewards.csv',
        index=False
    )


    # --------------------------------------------------------
    # Post-hoc only if Friedman significant
    # --------------------------------------------------------

    if not friedman_significant:

        print(
            "\nPost-hoc Wilcoxon tests were NOT performed."
        )

        print(
            "Reason: Friedman test was not statistically significant."
        )

        return {
            'friedman_stat': friedman_stat,
            'friedman_p': friedman_p,
            'posthoc': None
        }


    # ========================================================
    # PAIRWISE WILCOXON
    # ========================================================

    print("\n" + "=" * 80)
    print(
        f"{model_name}: POST-HOC WILCOXON + HOLM"
    )
    print("=" * 80)

    pair_results = []

    for method1, method2 in combinations(
        methods,
        2
    ):

        result = safe_wilcoxon(
            data[method1].values,
            data[method2].values
        )

        pair_results.append({

            'Model':
                model_name,

            'Method 1':
                method1,

            'Method 2':
                method2,

            'Statistic':
                result['statistic'],

            'Raw p-value':
                result['pvalue'],

            'Median Difference':
                result['median_difference'],

            'N':
                result['n_total'],

            'Non-zero differences':
                result['n_nonzero']

        })


    # ========================================================
    # HOLM CORRECTION
    # ========================================================

    raw_pvalues = [
        r['Raw p-value']
        for r in pair_results
    ]

    reject, p_adjusted, _, _ = multipletests(
        raw_pvalues,
        alpha=ALPHA,
        method='holm'
    )

    for i, result in enumerate(pair_results):

        result[
            'Holm adjusted p-value'
        ] = p_adjusted[i]

        result[
            'Significant after Holm'
        ] = reject[i]


    results_df = pd.DataFrame(
        pair_results
    )


    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        f'{model_name.lower()}_posthoc_wilcoxon_holm.csv',
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\nTotal pairwise comparisons:",
        len(results_df)
    )

    print(
        "Significant before Holm:",
        (
            results_df['Raw p-value'] < ALPHA
        ).sum()
    )

    print(
        "Significant after Holm:",
        results_df[
            'Significant after Holm'
        ].sum()
    )


    # ========================================================
    # SIGNIFICANT PAIRS
    # ========================================================

    significant = results_df[
        results_df[
            'Significant after Holm'
        ]
    ].copy()

    print("\n" + "-" * 80)
    print(
        f"{model_name}: SIGNIFICANT PAIRS AFTER HOLM"
    )
    print("-" * 80)

    if len(significant) > 0:

        print(
            significant[
                [
                    'Method 1',
                    'Method 2',
                    'Statistic',
                    'Raw p-value',
                    'Holm adjusted p-value',
                    'Median Difference',
                    'N'
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No pairwise comparison remained "
            "statistically significant after Holm correction."
        )


    # ========================================================
    # 10 SMALLEST RAW P-VALUES
    # ========================================================

    print("\n" + "-" * 80)
    print(
        f"{model_name}: 10 SMALLEST RAW P-VALUES"
    )
    print("-" * 80)

    smallest = results_df.sort_values(
        'Raw p-value'
    ).head(10)

    print(
        smallest[
            [
                'Method 1',
                'Method 2',
                'Raw p-value',
                'Holm adjusted p-value',
                'Median Difference'
            ]
        ].to_string(
            index=False
        )
    )


    return {
        'friedman_stat': friedman_stat,
        'friedman_p': friedman_p,
        'posthoc': results_df
    }


# ============================================================
# RUN FRIEDMAN FOR DQN
# ============================================================

dqn_results = run_friedman_analysis(
    dqn_data,
    'DQN'
)


# ============================================================
# RUN FRIEDMAN FOR PPO
# ============================================================

ppo_results = run_friedman_analysis(
    ppo_data,
    'PPO'
)


# ============================================================
# DQN vs PPO
# SAME REWARD FUNCTION
# ============================================================
#
# DQN-R0 vs PPO-R0
# DQN-R1 vs PPO-R1
# ...
# DQN-R13 vs PPO-R13
#
# 14 paired comparisons
# ============================================================

print("\n" + "=" * 80)
print("DQN vs PPO: SAME REWARD FUNCTION")
print("=" * 80)

print(
    "\nComparing DQN and PPO under the same reward function..."
)

cross_results = []


for i in range(N_REWARDS):

    dqn_name = f'DQN-R{i}'
    ppo_name = f'PPO-R{i}'

    x = dqn_common[dqn_name].values
    y = ppo_common[ppo_name].values

    result = safe_wilcoxon(
        x,
        y
    )

    cross_results.append({

        'Reward':
            f'R{i}',

        'DQN':
            dqn_name,

        'PPO':
            ppo_name,

        'Statistic':
            result['statistic'],

        'Raw p-value':
            result['pvalue'],

        'Median Difference (DQN-PPO)':
            result['median_difference'],

        'N':
            result['n_total'],

        'Non-zero differences':
            result['n_nonzero']

    })


# ============================================================
# HOLM FOR 14 DQN-PPO COMPARISONS
# ============================================================

cross_raw_pvalues = [
    r['Raw p-value']
    for r in cross_results
]

cross_reject, cross_adjusted, _, _ = multipletests(
    cross_raw_pvalues,
    alpha=ALPHA,
    method='holm'
)


for i, result in enumerate(cross_results):

    result[
        'Holm adjusted p-value'
    ] = cross_adjusted[i]

    result[
        'Significant after Holm'
    ] = cross_reject[i]


cross_results_df = pd.DataFrame(
    cross_results
)


# ============================================================
# SAVE DQN vs PPO RESULTS
# ============================================================

cross_results_df.to_csv(
    'dqn_vs_ppo_same_reward_wilcoxon_holm.csv',
    index=False
)


# ============================================================
# PRINT DQN vs PPO SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("DQN vs PPO SUMMARY")
print("-" * 80)

print(
    "Number of comparisons:",
    len(cross_results_df)
)

print(
    "Significant before Holm:",
    (
        cross_results_df['Raw p-value'] < ALPHA
    ).sum()
)

print(
    "Significant after Holm:",
    cross_results_df[
        'Significant after Holm'
    ].sum()
)


# ============================================================
# DQN vs PPO: ALL RESULTS
# ============================================================

print("\n" + "-" * 80)
print("DQN vs PPO: ALL 14 COMPARISONS")
print("-" * 80)

print(
    cross_results_df[
        [
            'Reward',
            'Statistic',
            'Raw p-value',
            'Holm adjusted p-value',
            'Median Difference (DQN-PPO)',
            'N'
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SIGNIFICANT DQN vs PPO RESULTS
# ============================================================

significant_cross = cross_results_df[
    cross_results_df[
        'Significant after Holm'
    ]
].copy()


print("\n" + "-" * 80)
print("DQN vs PPO: SIGNIFICANT RESULTS AFTER HOLM")
print("-" * 80)

if len(significant_cross) > 0:

    print(
        significant_cross[
            [
                'Reward',
                'DQN',
                'PPO',
                'Raw p-value',
                'Holm adjusted p-value',
                'Median Difference (DQN-PPO)',
                'N'
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No DQN-PPO comparison remained "
        "statistically significant after Holm correction."
    )


# ============================================================
# SAVE DAILY RETURNS
# ============================================================

dqn_data.to_csv(
    'dqn_14_rewards_daily_returns.csv',
    index=False
)

ppo_data.to_csv(
    'ppo_14_rewards_daily_returns.csv',
    index=False
)

benchmark_data.to_csv(
    'benchmark_daily_returns.csv',
    index=False
)


# ============================================================
# BENCHMARK DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("BENCHMARK DESCRIPTIVE STATISTICS")
print("=" * 80)

benchmark_descriptive = benchmark_data.describe().T[
    [
        'count',
        'mean',
        'std',
        'min',
        '50%',
        'max'
    ]
]

print(
    benchmark_descriptive.to_string()
)

benchmark_descriptive.to_csv(
    'benchmark_descriptive_statistics.csv'
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL STATISTICAL ANALYSIS SUMMARY")
print("=" * 80)

print("\n1. DQN - 14 Reward Functions")

print(
    f"Friedman χ² = "
    f"{dqn_results['friedman_stat']:.6f}"
)

print(
    f"df = 13"
)

print(
    f"p = "
    f"{format_pvalue(dqn_results['friedman_p'])}"
)


print("\n2. PPO - 14 Reward Functions")

print(
    f"Friedman χ² = "
    f"{ppo_results['friedman_stat']:.6f}"
)

print(
    f"df = 13"
)

print(
    f"p = "
    f"{format_pvalue(ppo_results['friedman_p'])}"
)


print("\n3. DQN vs PPO - Same Reward")

print(
    "14 paired Wilcoxon tests + Holm correction"
)

print(
    "Significant after Holm:",
    cross_results_df[
        'Significant after Holm'
    ].sum()
)


# ============================================================
# FILES GENERATED
# ============================================================

print("\n" + "=" * 80)
print("FILES GENERATED")
print("=" * 80)

print("""
DQN:
1. dqn_friedman_14_rewards.csv
2. dqn_posthoc_wilcoxon_holm.csv
3. dqn_14_rewards_descriptive_statistics.csv
4. dqn_14_rewards_daily_returns.csv

PPO:
5. ppo_friedman_14_rewards.csv
6. ppo_posthoc_wilcoxon_holm.csv
7. ppo_14_rewards_descriptive_statistics.csv
8. ppo_14_rewards_daily_returns.csv

DQN vs PPO:
9. dqn_vs_ppo_same_reward_wilcoxon_holm.csv

Benchmarks:
10. benchmark_descriptive_statistics.csv
11. benchmark_daily_returns.csv
""")

print("\nAnalysis completed successfully.")