import math
import numpy as np
import torch
from tqdm import tqdm
from xiangqi_rl.logger import logger

class MCTS:
    """Monte Carlo Tree Search implementation"""
    def __init__(self, model, num_simulations=800, c_puct=1.0, max_moves=200, disable_progress_bar=False):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_moves = max_moves  # Maximum moves per game
        self.disable_progress_bar = disable_progress_bar
        self.Qsa = {}  # stores Q values for state-action pairs
        self.Nsa = {}  # stores visit counts for state-action pairs
        self.Ns = {}   # stores visit count for states
        self.Ps = {}   # stores initial policy (returned by neural network)
        self.policy_cache = {}  # Cache for policy predictions
        
    def _move_to_index(self, move):
        """Convert move tuple ((from_row, from_col), (to_row, to_col)) to index"""
        (from_row, from_col), (to_row, to_col) = move
        return from_row * 9 * 90 + from_col * 90 + to_row * 9 + to_col

    def _index_to_move(self, index):
        """Convert index to move tuple"""
        from_row = index // (9 * 90)
        from_col = (index % (9 * 90)) // 90
        to_row = (index % 90) // 9
        to_col = index % 9
        return ((from_row, from_col), (to_row, to_col))

    def clear_tree(self):
        """Clear the search tree and cache to free memory"""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.policy_cache.clear()

    def search(self, env, debug=False):
        """Perform MCTS search"""
        # Clear tree before each search to prevent memory buildup
        self.clear_tree()
        
        s = env.get_canonical_state()
        
        # Add progress bar for simulations
        pbar = None if self.disable_progress_bar else tqdm(range(self.num_simulations), desc="MCTS", leave=False)
        for i in pbar if pbar else range(self.num_simulations):
            env_copy = env.clone()
            if pbar and i % 10 == 0:
                pbar.set_description(f"MCTS sim {i}/{self.num_simulations} Q:{self.Qsa.get((s,0), 0):.2f}")
            self._simulate(env_copy, s, move_count=0, position_history=set())
            
        valid_moves = env.get_valid_moves()
        if debug:
            self.print_mcts_stats(env, s, valid_moves)
        
        # Calculate move probabilities
        move_probs = np.zeros(self.model.policy_head[-1].out_features)
        
        # Log the visit counts for valid moves
        total_visits = 0
        for move in valid_moves:
            move_idx = self._move_to_index(move)
            visits = self.Nsa.get((s, move_idx), 0)
            move_probs[move_idx] = visits
            total_visits += visits
            if visits > 0:
                logger.debug(f"Move {move}: {visits} visits ({visits/total_visits*100:.1f}%)")
        
        # Normalize probabilities
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()
            
        # Add logging for search completion
        logger.debug(f"MCTS search completed. Total visits: {total_visits}")
        return move_probs
    
    def _simulate(self, env, state, move_count=0, position_history=None):
        """Simulate one instance of MCTS"""
        if position_history is None:
            position_history = set()
            
        # Early stopping for clearly won/lost positions
        if move_count > 5:  # Only check after a few moves
            if abs(env.get_reward()) == 1:  # Clear win/loss
                return -env.get_reward()
                
        # Check for repetition or max moves
        current_position = env.get_canonical_state()
        if current_position in position_history or move_count >= self.max_moves:
            return 0.0  # Draw
            
        position_history.add(current_position)
        
        if state not in self.Ps:
            # Use cached policy if available
            if state in self.policy_cache:
                self.Ps[state] = self.policy_cache[state]
                return -self.policy_cache[state][1]  # Return cached value
                
            # Leaf node - evaluate position
            state_array = np.array([env.get_canonical_state()])
            # Reshape state array to match expected dimensions
            state_array = state_array.reshape(1, 14, 10, 9)
            state_tensor = torch.FloatTensor(state_array).to(self.model.device)
            
            with torch.no_grad():
                policy, value = self.model(state_tensor)
                # Only move to CPU if we're not using GPU
                if self.model.device == 'cpu':
                    policy = torch.softmax(policy[0], dim=0).cpu().detach().numpy()
                else:
                    # Keep on GPU for faster operations
                    policy = torch.softmax(policy[0], dim=0).detach()
                    # Convert to numpy only when needed for specific operations
                    policy = policy.cpu().numpy()
                value = value.item()
            
            # Cache the results
            self.Ps[state] = policy
            self.policy_cache[state] = (policy, value)
            
            # Clear cache if too large
            if len(self.policy_cache) > 10000:
                self.policy_cache.clear()
                
            return -value
            
        # Select action with highest UCB score
        valid_moves = env.get_valid_moves()
        if not valid_moves:  # If no valid moves, treat as game over
            return -env.get_reward()
            
        cur_best = float('-inf')
        best_act = None
        
        for move in valid_moves:
            move_idx = self._move_to_index(move)
            
            q = self.Qsa.get((state, move_idx), 0)
            n = self.Nsa.get((state, move_idx), 0)
            p = self.Ps[state][move_idx]
            
            ucb = q + self.c_puct * p * math.sqrt(self.Ns.get(state, 0)) / (1 + n)
            
            if ucb > cur_best:
                cur_best = ucb
                best_act = move
                best_idx = move_idx
                
        env.step(best_act)
        v = self._simulate(env, env.get_canonical_state(), move_count + 1, position_history)
        
        # Update statistics
        if (state, best_idx) in self.Qsa:
            self.Qsa[(state, best_idx)] = (self.Nsa[(state, best_idx)] * self.Qsa[(state, best_idx)] + v) / (self.Nsa[(state, best_idx)] + 1)
            self.Nsa[(state, best_idx)] += 1
        else:
            self.Qsa[(state, best_idx)] = v
            self.Nsa[(state, best_idx)] = 1
            
        self.Ns[state] = self.Ns.get(state, 0) + 1
        return -v

    def print_mcts_stats(self, env, s, valid_moves):
        """Print MCTS statistics for debugging"""
        print("\nMCTS Statistics:")
        print(f"Current player: {'Red' if env.current_player else 'Black'}")
        print("\nTop moves by visit count:")
        
        # Collect move stats
        move_stats = []
        for move in valid_moves:
            move_idx = self._move_to_index(move)
            visits = self.Nsa.get((s, move_idx), 0)
            q_value = self.Qsa.get((s, move_idx), 0)
            prior = self.Ps[s][move_idx] if s in self.Ps else 0
            
            move_stats.append({
                'move': move,
                'visits': visits,
                'Q': q_value,
                'P': prior,
                'N': visits
            })
        
        # Sort by visit count
        move_stats.sort(key=lambda x: x['visits'], reverse=True)
        
        # Print top 5 moves
        for i, stat in enumerate(move_stats[:5]):
            print(f"{i+1}. Move {stat['move']}: "
                  f"N={stat['visits']} "
                  f"Q={stat['Q']:.3f} "
                  f"P={stat['P']:.3f}")