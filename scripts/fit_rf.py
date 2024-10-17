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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# %%
# load data
train_df = pl.read_csv(upstream["filter"]["train"])
test_df = pl.read_csv(upstream["filter"]["test"])
train_df = train_df.drop("engine_id", "cycles", "rul")
test_df = test_df.drop("engine_id", "cycles", "rul")

y_train = train_df["rul_clip"]
x_train = train_df.drop("rul_clip")
y_test = test_df["rul_clip"]
x_test = test_df.drop("rul_clip")

# %%
# Convert to nparray
y_train = y_train.to_numpy()
x_train = x_train.to_numpy()
y_test = y_test.to_numpy()
x_test = x_test.to_numpy()

# %%
# Random Forest
rf = RandomForestRegressor(n_estimators=250, max_features="sqrt",
                           n_jobs=-1, random_state=26)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)

rmse = root_mean_squared_error(y_test, pred)
variance = r2_score(y_test, pred)

print(f'test set RMSE: {rmse:.4f}, R2: {variance:.4f}')
