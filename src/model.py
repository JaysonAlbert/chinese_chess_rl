import torch
import torch.nn as nn
import torch.nn.functional as F

class XiangqiHybridNet(nn.Module):
    def __init__(self):
        super(XiangqiHybridNet, self).__init__()
        
        # Input channels: 14 (7 piece types * 2 colors)
        # Board size: 10 x 9
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 10 * 9, 90 * 90)  # All possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 10 * 9)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 10 * 9)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
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