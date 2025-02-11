import torch
import numpy as np
from xiangqi_rl.train import MCTS

class XiangqiAgent:
    def __init__(self, model, env, num_simulations=100):
        self.model = model
        self.env = env  # Use the environment passed from AIPlayer
        self.mcts = MCTS(model, num_simulations=num_simulations)
    
    def select_action(self, valid_moves, temperature=1.0):
        """Select an action using MCTS search"""
        with torch.no_grad():
            # Get move probabilities from MCTS
            pi = self.mcts.search(self.env)
            
            # Filter for valid moves only
            valid_move_probs = []
            for move in valid_moves:
                move_idx = self.mcts._move_to_index(move)
                valid_move_probs.append(pi[move_idx])
            
            # Apply temperature
            if temperature != 1.0:
                probs = [p ** (1/temperature) for p in valid_move_probs]
            else:
                probs = valid_move_probs
                
            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
                
                # Select move based on probabilities
                move_idx = np.random.choice(len(valid_moves), p=probs)
                return valid_moves[move_idx]
            
            # Fallback to random move if something goes wrong
            return np.random.choice(valid_moves)
    
    def _move_to_index(self, move):
        """Convert a move (from_pos, to_pos) to a single index"""
        from_pos, to_pos = move
        from_idx = from_pos[0] * 9 + from_pos[1]
        to_idx = to_pos[0] * 9 + to_pos[1]
        return from_idx * 90 + to_idx 