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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

# %%
# load data
train_df = pl.read_csv(upstream["filter"]["train"])
test_df = pl.read_csv(upstream["filter"]["test"]).group_by("engine_id", maintain_order=True).last()
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
# LSTM


class BiLSTMModel(nn.Module):
    def __init__(self, input_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 50, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(50*2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# %%
# Prepare data for LSTM


scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)
x_train_sc = x_train_sc.reshape((x_train_sc.shape[0], 1, x_train_sc.shape[1]))
x_test_sc = x_test_sc.reshape((x_test_sc.shape[0], 1, x_test_sc.shape[1]))

# %%
# Training

X_train_tensor = torch.FloatTensor(x_train_sc)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(x_test_sc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

lstm = BiLSTMModel(x_train_sc.shape[2]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters())

epochs = 25
for epoch in range(epochs):
    lstm.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = lstm(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    lstm.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = lstm(batch_X)
            val_loss += criterion(outputs.squeeze(), batch_y).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# %%
# Evaluation


def evaluate(y_true, y_hat, label='test'):
    rmse = root_mean_squared_error(y_true, y_hat)
    variance = r2_score(y_true, y_hat)
    print(f'{label} set RMSE: {rmse:.4f}, R2: {variance:.4f}')
    return rmse, variance


lstm.eval()
with torch.no_grad():
    y_hat_test = lstm(X_test_tensor.to(device)).cpu().numpy().squeeze()


evaluate(y_test, y_hat_test, 'test')


