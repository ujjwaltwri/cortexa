import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# --- Configuration ---
SEQUENCE_LENGTH = 20  # Look back 20 days
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # The LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # The Output Layer (Predicts 0 or 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward Pass
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the output of the LAST time step
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

def create_sequences(data, target, seq_length):
    """Converts 2D data into 3D sequences (Samples, Time, Features)"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_neural_model(config_path="config.yaml"):
    print("--- 🧠 Cortexa 3.0: Initializing Neural Training (LSTM) ---")
    
    # 1. Load Data
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # 2. Preprocessing (Neural Nets NEED Scaled Data)
    exclude_cols = ['target', 'future_close', 'future_return', 'ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Scale features to 0-1 range (Critical for LSTMs)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 3. Create Sequences
    # We group by Ticker to ensure we don't sequence data across different stocks
    print(f"Sequencing data (Window: {SEQUENCE_LENGTH} days)...")
    
    X_list, y_list = [], []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        if len(ticker_df) < SEQUENCE_LENGTH + 10: continue
        
        data = ticker_df[feature_cols].values
        target = ticker_df['target'].values
        
        X_seq, y_seq = create_sequences(data, target, SEQUENCE_LENGTH)
        X_list.append(X_seq)
        y_list.append(y_seq)
        
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    # 4. Train/Test Split (Time-based isn't perfect here due to multi-ticker shuffle, 
    # but we'll use simple split for the prototype)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to Tensors
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=len(feature_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training on {device}...")
    
    # 6. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    # 7. Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    binary_preds = (all_preds > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    print("\n--- 🧠 Neural Results ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC:  {auc:.4f}")
    
    # 8. Save Artifacts
    save_dir = Path(config["ml_models"]["saved_models"])
    torch.save(model.state_dict(), save_dir / "lstm_model.pth")
    joblib.dump(scaler, save_dir / "lstm_scaler.pkl") # We MUST save the scaler
    
    metadata = {
        "features": feature_cols,
        "metrics": {"accuracy": acc, "roc_auc": auc},
        "threshold": 0.5,
        "model_type": "lstm",
        "sequence_length": SEQUENCE_LENGTH
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)

    print("✅ Neural Brain Saved.")

if __name__ == "__main__":
    train_neural_model()