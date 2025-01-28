import torch
import numpy as np

class XiangqiAgent:
    def __init__(self, model):
        self.model = model
    
    def select_action(self, state, valid_moves, temperature=1.0):
        """Select an action using the model's policy"""
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
            
            # Get policy from model
            policy, _ = self.model(state_tensor)
            policy = policy.squeeze(0)
            
            # Convert valid moves to indices
            valid_indices = [self._move_to_index(move) for move in valid_moves]
            
            # Mask invalid moves
            mask = torch.zeros_like(policy)
            mask[valid_indices] = 1
            policy = policy * mask
            
            # Apply temperature
            if temperature != 1.0:
                policy = torch.pow(policy, 1.0/temperature)
            
            # Normalize probabilities
            policy = policy / policy.sum()
            
            # Sample action
            action_idx = torch.multinomial(policy, 1).item()
            
            # Convert back to board coordinates
            for move in valid_moves:
                if self._move_to_index(move) == action_idx:
                    return move
            
            # Fallback to first valid move if something goes wrong
            return valid_moves[0]
    
    def _move_to_index(self, move):
        """Convert a move (from_pos, to_pos) to a single index"""
        from_pos, to_pos = move
        from_idx = from_pos[0] * 9 + from_pos[1]
        to_idx = to_pos[0] * 9 + to_pos[1]
        return from_idx * 90 + to_idx 