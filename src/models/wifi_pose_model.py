import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class WiFiPoseModel(nn.Module):
    def __init__(self, in_channels=9, subcarriers=30, time_steps=100, 
                 hidden_dim=128, num_layers=4, num_heads=4, 
                 dropout=0.1, num_joints=14, grad_checkpointing=True):
        super(WiFiPoseModel, self).__init__()
        self.grad_checkpointing = grad_checkpointing
        self.num_joints = num_joints
        
        # 2D CNN Feature Extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=(1, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=3, stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.relu3 = nn.ReLU()
        
        # Calculate sequence length after CNN
        # subcarriers (30) -> /1 (stride 1) -> 30 -> /2 (stride 2) -> 15 -> /2 (stride 2) -> 8
        # time_steps (100) -> /2 (stride 2) -> 50 -> /2 (stride 2) -> 25 -> /2 (stride 2) -> 13
        # Global pooling over subcarrier dimension to get sequence representation
        self.pool = nn.AdaptiveAvgPool2d((1, None)) # Output: (batch, hidden_dim, 1, seq_len)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 13, hidden_dim))
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression MLP Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 13, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_joints * 3)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_encoder, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x shape: (batch_size, C, H, W) = (batch_size, 9, 30, 100)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        
        out = self.pool(out) # shape: (batch, hidden_dim, 1, seq_len=13)
        out = out.squeeze(2).permute(0, 2, 1) # shape: (batch, 13, hidden_dim)
        
        # Add positional encoding
        out = out + self.pos_encoder
        
        # Transformer encoding (with optional gradient checkpointing)
        if self.grad_checkpointing and self.training:
            # Wrap transformer execution in checkpointing
            out = checkpoint.checkpoint(self.transformer_encoder, out, use_reentrant=False)
        else:
            out = self.transformer_encoder(out)
            
        # Flatten and regress
        out = out.reshape(out.size(0), -1) # shape: (batch, 13 * hidden_dim)
        pred_pose = self.fc(out) # shape: (batch, 14 * 3)
        
        # Reshape to (batch, 14, 3)
        pred_pose = pred_pose.view(-1, self.num_joints, 3)
        return pred_pose
