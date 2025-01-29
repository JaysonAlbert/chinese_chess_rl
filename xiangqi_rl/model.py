import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    """Get the best available device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x

class XiangqiHybridNet(nn.Module):
    def __init__(self, num_res_blocks=19, num_channels=256):
        super(XiangqiHybridNet, self).__init__()
        
        # Input channels: 14 (7 piece types * 2 colors)
        # Board size: 10 x 9
        
        # Initial processing
        self.conv_input = nn.Sequential(
            nn.Conv2d(14, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 9, 90 * 90)  # All possible moves
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Initial convolution
        x = self.conv_input(x)
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

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
        """Override to method to handle device transfer properly"""
        self.device = device
        return super().to(device) 