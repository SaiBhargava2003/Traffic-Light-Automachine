import torch
import torch.nn as nn

# Define LSTM model
class TrafficPredictor(nn.Module):
    def __init__(self):
        super(TrafficPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=X.shape[2], hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, X.shape[2])

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model, define loss and optimizer
model = TrafficPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
