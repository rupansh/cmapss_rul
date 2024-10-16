# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, declare them in the YAML spec and leave this
# as None
upstream = None

# This is a placeholder, leave it as None
product = None


# %%
# imports
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.sensors import SENSORS

# %%
# Load Data
train_df = pl.read_csv(upstream["filter"]["train"])
train_pd = train_df.to_pandas()

# %%
# Setup seaborn
sns.set_theme()

# %%
# Plot sensor vs RUL


def plot_sensor(name):
    plt.figure(figsize=(13, 5))
    for i in train_pd['engine_id'].unique():
        if i % 10 == 0:
            sns.lineplot(x='rul', y=name,
                         data=train_pd[train_pd['engine_id'] == i])
    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(name)
    plt.xlabel('Remaining Useful Life')
    plt.show()


for sensor in SENSORS:
    if sensor not in train_df.columns:
        continue
    plot_sensor(sensor)

# %%
# Maximum Cycles for each engine unit
max_time_cycles = train_df.group_by("engine_id").agg(pl.max("cycles"))
plt.figure(figsize=(20, 50))
max_time_cycles = max_time_cycles.to_pandas()
ax = sns.barplot(x="cycles", y="engine_id", data=max_time_cycles, orient='h', width=0.8)
ax.set_title('Engine Lifetime', fontweight='bold', size=30)
ax.set_xlabel('Cycles', fontweight='bold', size=20)
ax.set_ylabel('Engine ID', fontweight='bold', size=20)
ax.tick_params(axis='x', labelsize=15)

ax.tick_params(axis='y', labelsize=15)
ax.grid(True)
sns.despine()
plt.show()

