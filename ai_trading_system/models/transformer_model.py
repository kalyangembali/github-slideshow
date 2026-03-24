import torch
import torch.nn as nn
import math
from loguru import logger
import os

from config import settings

class TransformerPredictor(nn.Module):
    """
    Phase 5: Transformer attention model
    Input: 60 candles x 55 features
    Output: BUY/SELL/HOLD + probability
    """
    def __init__(self, input_dim=55, d_model=64, num_heads=4, num_layers=2, num_classes=3, seq_len=60):
        super(TransformerPredictor, self).__init__()

        # Projection layer to match the dimension expected by Transformer (d_model)
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional Encoding to inject sequence order info
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output Dense Layers
        self.fc1 = nn.Linear(d_model * seq_len, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, src):
        # src shape: (batch_size, seq_len=60, features=55)
        src = self.input_projection(src) # (batch, seq, d_model)
        src = self.pos_encoder(src)      # Add positional encoding

        output = self.transformer_encoder(src) # (batch, seq, d_model)

        # Flatten for dense layers
        output = output.view(output.size(0), -1) # (batch, seq * d_model)

        x = self.relu(self.fc1(output))
        return self.out(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)
        # Add pe tensor matching the seq_len
        x = x + self.pe[:x.size(1), :]
        return x

def save_model(model, name="transformer_model.pth"):
    model_dir = os.path.join(settings.BASE_DIR, "models", "saved")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, name)
    torch.save(model.state_dict(), path)
    logger.info(f"Transformer Model saved to {path}")
