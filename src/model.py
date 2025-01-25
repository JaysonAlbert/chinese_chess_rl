import torch
import torch.nn as nn
import torch.nn.functional as F

class XiangqiHybridNet(nn.Module):
    def __init__(self, channels=256, num_attention_heads=8):
        super(XiangqiHybridNet, self).__init__()
        
        # Use torch.nn.LazyLinear for automatic input size inference
        self.conv_layers = nn.Sequential(
            nn.Conv2d(14, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # Use inplace operations
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Position encoding for attention
        self.register_buffer('pos_encoding', torch.randn(1, 90, channels))  # 90 = 9x10 board
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 90, 1024),
            nn.ReLU(),
            nn.Linear(1024, 90 * 90),  # All possible moves
            nn.LogSoftmax(dim=1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 90, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)  # Shape: [batch, channels, 10, 9]
        
        # Prepare for attention
        features = conv_features.view(batch_size, conv_features.size(1), -1)  # [batch, channels, 90]
        features = features.permute(0, 2, 1)  # [batch, 90, channels]
        
        # Add positional encoding
        features = features + self.pos_encoding
        
        # Self-attention
        attended_features, _ = self.attention(features, features, features)
        
        # Reshape back to spatial form
        attended_features = attended_features.permute(0, 2, 1)  # [batch, channels, 90]
        attended_features = attended_features.view(batch_size, -1, 10, 9)
        
        # Combine CNN and attention features
        final_features = attended_features + conv_features  # Residual connection
        
        # Policy output
        policy = self.policy_head(final_features)
        
        # Value output
        value = self.value_head(final_features)
        
        return policy, value

    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        batch_size = x.size(0)
        
        with torch.no_grad():
            # CNN feature extraction
            conv_features = self.conv_layers(x)
            features = conv_features.view(batch_size, conv_features.size(1), -1)
            features = features.permute(0, 2, 1)
            features = features + self.pos_encoding
            
            # Get attention weights
            _, attention_weights = self.attention(features, features, features)
            
            return attention_weights

    def to(self, device):
        """Override to() to ensure pos_encoding moves to the correct device"""
        super().to(device)
        return self 