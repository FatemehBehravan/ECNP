import torch
import torch.nn as nn

# bidirectional LSTM

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, lstm_hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        
        # LSTM دوطرفه
        self.lstm = nn.LSTM(
            input_size=input_size,  # context_x (1) + context_y (1) = 2
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # فعال برای بهبود extrapolation
            dropout=dropout if num_layers > 1 else 0
        )
        
        # LayerNorm برای پایداری
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)  # *2 به دلیل bidirectional
        
        # مکانیزم توجه
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # *2 به دلیل bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # لایه‌های خطی عمیق‌تر
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # ورودی: 128، خروجی: 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, lstm_hidden_size)  # خروجی: 64
        )
        
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, context_x, context_y, target_x):
        # ترکیب context_x و context_y
        x = torch.cat([context_x, context_y], dim=-1)  # (batch_size, seq_len, 2)
        
        # پردازش با LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, lstm_hidden_size * 2)
        
        # اعمال LayerNorm
        lstm_out = self.layer_norm(lstm_out)  # (batch_size, seq_len, lstm_hidden_size * 2)
        
        # اعمال توجه
        attn_output, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out  # query, key, value
        )  # attn_output: (batch_size, seq_len, lstm_hidden_size * 2)
        
        # میانگین وزن‌دار خروجی‌های توجه
        out = attn_output.mean(dim=1)  # (batch_size, lstm_hidden_size * 2)
        
        # پردازش نهایی با لایه‌های خطی
        out = self.fc(out)  # (batch_size, lstm_hidden_size)
        
        return out













# Simple model

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, lstm_hidden_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.lstm_hidden_size = lstm_hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(
#             input_size=input_size,  # context_x (1) + context_y (1) = 2
#             hidden_size=lstm_hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.fc = nn.Linear(lstm_hidden_size, lstm_hidden_size)

#         # Initialize weights
#         for name, param in self.named_parameters():
#             if 'weight' in name and param.dim() >= 2:
#                 nn.init.xavier_uniform_(param)
#             elif 'bias' in name:
#                 nn.init.zeros_(param)

#     def forward(self, context_x, context_y, target_x):
#         # Combine context_x and context_y
#         x = torch.cat([context_x, context_y], dim=-1)  # (batch, seq_len, 2)
#         # LSTM processing
#         out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, lstm_hidden_size)
#         # Take the last timestep
#         out = out[:, -1, :]  # (batch, lstm_hidden_size)
#         out = self.fc(out)  # (batch, lstm_hidden_size)
#         return out