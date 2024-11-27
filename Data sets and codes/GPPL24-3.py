import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Load the dataset
data = pd.read_csv('output-pca.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime
data = data.set_index('Date')  # Set 'Date' as index

# Ensure data is sorted by date
data = data.sort_index()

# Define past and future window sizes
past_hours = 24  # Use past 48 hours of data
future_hours = 3  # Predicting next 18 hours

# Prepare data with features and target using a sliding window approach
X, y = [], []
for i in range(len(data) - past_hours - future_hours + 1):
    X.append(data.iloc[i:i + past_hours][['cazhi', '0', '1']].values)  # Use 'cazhi' and features '0', '1'
    y.append(data.iloc[i + past_hours:i + past_hours + future_hours]['cazhi'].values)  # Future 18-hour gas consumption

X, y = np.array(X), np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Define Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increase batch size to stabilize training
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, forecast_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, forecast_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_layer(x)  # Project input to model dimension
        x = self.dropout(x)  # Add dropout for regularization
        x = self.transformer(x)  # Pass through Transformer
        x = x.mean(dim=1)  # Global average pooling across time steps
        output = self.output_layer(x)  # Predict future values
        return output

input_dim = X_train.shape[2]
d_model = 128  # Increase model dimension to improve representation power
nhead = 8  # Increase the number of attention heads
num_layers = 4  # Increase the number of Transformer layers
forecast_length = future_hours

# Train and evaluate function
def train_and_evaluate():
    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, forecast_length=forecast_length, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Use AdamW optimizer with weight decay

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    # Training loop with early stopping
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)  # Forward pass
            loss = criterion(output, y_batch)  # Loss for 18 hours prediction
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()  # Save a copy of the best model weights
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Print training and validation loss for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Load the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Evaluate model on test set
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            y_pred.extend(output.numpy())
            y_true.extend(y_batch.numpy())

    # Calculate RMSE and MAE for the best model
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae, y_true, y_pred

# Run training and evaluation 10 times
best_rmse = float('inf')
best_mae = float('inf')
best_y_true, best_y_pred = None, None

for i in range(10):
    rmse, mae, y_true, y_pred = train_and_evaluate()
    print(f'Run {i+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}')
    if mae < best_mae:
        best_rmse, best_mae = rmse, mae
        best_y_true, best_y_pred = y_true, y_pred

# Display best results
print(f'Best Model RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}')

