# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from google.colab import files

# Upload and read the dataset
uploaded = files.upload()
df = pd.read_hdf('METR-LA.h5')

# Scale the data
traffic_data = df.values
scaler = MinMaxScaler()
traffic_data_scaled = scaler.fit_transform(traffic_data)

# Prepare the dataset with time_steps (past data for prediction)
X, y = [], []
time_steps = 10  # Past 10 timesteps used for prediction
for i in range(time_steps, len(traffic_data_scaled)):
    X.append(traffic_data_scaled[i-time_steps:i])
    y.append(traffic_data_scaled[i])

X, y = np.array(X), np.array(y)

# Split the dataset into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
