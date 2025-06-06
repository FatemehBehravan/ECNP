import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.output_projection = nn.Linear(output_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, context_x, context_y, target_x):
        # print('context_x.shape=', context_x.shape) # torch.Size([1, 50, 10, 4])
        # print('context_y.shape=', context_y.shape) # torch.Size([1, 50, 10, 1])
        # print('target_x.shape=', target_x.shape) # torch.Size([1, 58, 10, 4])
        
        # ترکیب context_x و context_y در محور ویژگی‌ها
        x = torch.cat([context_x, context_y], dim=-1)
        # print('x.shape=', x.shape) # torch.Size([1, 50, 10, 5])
        
        # مسطح کردن محور batch_size و num_points برای سازگاری با nn.Linear
        batch_size, num_points, seq_len, features = x.shape
        x = x.view(batch_size * num_points, seq_len, features)  # (1 * 50, 21, 5) = (50, 21, 5)
        
        # تبدیل به hidden_size
        x = self.input_projection(x)  # (50, 21, 64)
        # print('x.shape after input_projection=', x.shape)
        
        # transformer_encoder انتظار (batch_size, seq_len, feature_dim) دارد
        transformer_out = self.transformer_encoder(x)  # (50, 21, 64)
        
        # بازگرداندن به شکل اصلی (batch_size, num_points, seq_len, hidden_size)
        transformer_out = transformer_out.view(batch_size, num_points, seq_len, self.hidden_size)  # (1, 50, 21, 64)
        # print('transformer_out=', transformer_out.shape)

        # نرمال‌سازی و میانگین‌گیری روی محور seq_len
        out = self.layer_norm(transformer_out)
        
        # پردازش نهایی
        rep = self.fc(out)  # (1, 50, 21, 64)
        

        x = target_x
        batch_size, num_points, seq_len, features = x.shape
        x = x.view(batch_size * num_points, seq_len, features) # ([95, 10, 4])
        x = self.output_projection(x)  # torch.Size([95, 10, 64])
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.view(batch_size, num_points, seq_len, self.hidden_size)  # (1, 50, 21, 64)
        # print('transformer_out=', transformer_out.shape) # torch.Size([1, 94, 10, 64])
        out = self.layer_norm(transformer_out)
        target_x_rep = self.fc(out) # ([1, 94, 10, 64])

        return rep , target_x_rep
        