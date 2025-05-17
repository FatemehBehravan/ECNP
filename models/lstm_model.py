import torch
import torch.nn as nn

# bidirectional LSTM

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, lstm_hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(lstm_hidden_size, lstm_hidden_size)

        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, context_x, context_y, target_x):
        x = torch.cat([context_x, context_y], dim=-1)
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
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