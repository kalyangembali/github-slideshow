import torch
import torch.nn as nn
from loguru import logger
import os

from config import settings

class LSTMPredictor(nn.Module):
    """
    LSTM sequence model (PyTorch) as defined in Phase 5
    Architecture: LSTM(128) -> LSTM(64) -> Attention -> Dense(32) -> Output
    """
    def __init__(self, input_dim=55, hidden_1=128, hidden_2=64, num_classes=3):
        super(LSTMPredictor, self).__init__()

        # We need batch_first=True since our input is (batch, seq, feature)
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_1, hidden_size=hidden_2, batch_first=True)

        # Attention Layer
        self.attention = nn.Linear(hidden_2, 1)

        # Dense layers
        self.fc1 = nn.Linear(hidden_2, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # x is expected to be of shape (batch, seq_len=60, features=55)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        # Attention over the sequence dimension
        # out shape: (batch, seq_len, hidden_2)
        attn_weights = torch.softmax(self.attention(out), dim=1) # (batch, seq, 1)

        # Apply attention weights and sum over sequence length
        context_vector = torch.sum(attn_weights * out, dim=1) # (batch, hidden_2)

        # Dense -> Output
        x = self.relu(self.fc1(context_vector))
        return self.out(x) # We don't use softmax here since CrossEntropyLoss applies it

def save_model(model, name="lstm_model.pth"):
    model_dir = os.path.join(settings.BASE_DIR, "models", "saved")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, name)
    torch.save(model.state_dict(), path)
    logger.info(f"LSTM Model saved to {path}")
