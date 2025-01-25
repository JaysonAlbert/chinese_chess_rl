import torch
import numpy as np
from model import XiangqiHybridNet

class XiangqiAgent:
    def __init__(self, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else XiangqiHybridNet().to(self.device)
        self.model.eval()
    
    def select_action(self, state, valid_moves, temperature=1.0):
        """Select an action from valid moves using the model's policy"""
        # Move state tensor to the same device as the model
        device = next(self.model.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy, value = self.model(state_tensor)
            
            # Convert policy to probabilities
            policy = policy.exp().cpu().numpy()[0]
            
            # Mask invalid moves
            valid_move_mask = np.zeros_like(policy)
            for move in valid_moves:
                valid_move_mask[self._move_to_index(move)] = 1
            
            policy *= valid_move_mask
            policy /= policy.sum()
            
            # Sample move based on policy
            if temperature == 0:
                move_idx = policy.argmax()
            else:
                policy = policy ** (1/temperature)
                policy /= policy.sum()
                move_idx = np.random.choice(len(policy), p=policy)
            
            return self._index_to_move(move_idx)
    
    def _move_to_index(self, move):
        """Convert move coordinates to flat index"""
        from_pos, to_pos = move
        return from_pos[0] * 9 * 90 + from_pos[1] * 90 + to_pos[0] * 9 + to_pos[1]
    
    def _index_to_move(self, index):
        """Convert flat index to move coordinates"""
        from_row = index // (9 * 90)
        from_col = (index % (9 * 90)) // 90
        to_row = (index % 90) // 9
        to_col = index % 9
        return ((from_row, from_col), (to_row, to_col)) 