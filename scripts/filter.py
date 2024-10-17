# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
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
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None

# The engine series to load
series = None

# %%
# imports
import polars as pl
from src.sensors import SENSORS

# %%
# Load data

columns = [
    'engine_id',
    'cycles',
    'mach_nr',
    'altitude',
    'sea_lvl_temp',
] + SENSORS


def load_data(name: str, cols: list[str]):
    df = pl.read_csv(f"./data/{name}.txt", separator=' ', has_header=False, new_columns=cols)
    return df[[s.name for s in df if not (s.null_count() == df.height)]]


x_train = load_data(f"train_FD00{series}", columns)
x_test = load_data(f"test_FD00{series}", columns)
y_test = load_data(f"RUL_FD00{series}", ["max_rul"])

# %%
# Merge x_test & y_test
y_test = y_test.with_row_index(name="engine_id", offset=10001)
y_test = y_test.with_columns(pl.col("engine_id").cast(pl.Int64))

# add to engine id so they don't conflict with the train set
x_test = x_test.with_columns((pl.col("engine_id") + 10000).alias("engine_id"))
test_df = x_test.join(y_test, on="engine_id")
max_cycles = test_df.group_by("engine_id").agg(pl.max("cycles")).rename({"cycles": "max_cycles"})
test_df = test_df.join(max_cycles, on="engine_id")
test_df = test_df.with_columns(
    (pl.col("max_rul") + pl.col("max_cycles") - pl.col("cycles"))
    .alias("rul")
).drop(["max_rul", "max_cycles"])
test_df = test_df.group_by("engine_id", maintain_order=True).last()

# %%
# Derive y for train set
y_train = x_train.group_by("engine_id").agg(pl.max("cycles")).rename({"cycles": "max_rul"})

train_df = x_train.join(y_train, on="engine_id")
train_df = train_df.with_columns(
    (pl.col("max_rul") - pl.col("cycles"))
    .alias("rul")
).drop(["max_rul"])
train_df.select("engine_id", "cycles", "rul")

# %%
# Drop Constant features


def constant_features(df: pl.DataFrame):
    std_df = df.select(pl.selectors.numeric().std() <= 0.02)
    return std_df.unpivot().filter(pl.col("value")).to_series(0)


to_drop = constant_features(train_df)
train_df = train_df.drop(*to_drop)
test_df = test_df.drop(*to_drop)
train_df.head()

print("Constant features", to_drop)

# %%
# Keep features with strong correlation
corr = train_df.corr()
mask = corr.select(pl.col("rul").abs() < 0.5).to_series(0)
# Drop all columns, but keep engine id and cycles
to_drop = [column for (i, column) in enumerate(corr.columns) if i > 1 and mask[i]]
train_df = train_df.drop(*to_drop)
test_df = test_df.drop(*to_drop)
train_df.head()

print("Redundant columns", to_drop)

train_df.write_csv(product["train"])
test_df.write_csv(product["test"])
