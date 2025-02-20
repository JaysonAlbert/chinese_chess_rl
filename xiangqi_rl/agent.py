import torch
import numpy as np
from xiangqi_rl.mcts import MCTS
from xiangqi_rl.environment import XiangqiEnv
from xiangqi_rl.visualize import XiangqiVisualizer
import pygame
from xiangqi_rl.visualize import XiangqiVisualizer
import gc
from xiangqi_rl.logger import logger
from tqdm import tqdm
import os

class XiangqiAgent:
    def __init__(self, model, env, num_simulations=100, max_moves=200, show_board=False, disable_progress_bar=False):
        self.model = model
        self.env = env  # Use the environment passed from AIPlayer
        self.mcts = MCTS(model, num_simulations=num_simulations, max_moves=max_moves)
        self.show_board = show_board
        self.disable_progress_bar = disable_progress_bar
        self.max_moves = max_moves
        self.game_id = 0
        # Create games directory if it doesn't exist
        self.games_dir = "logs/games"
        os.makedirs(self.games_dir, exist_ok=True)
        
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
    
    def self_play(self):
        """Execute one episode of self-play"""
        logger.info("Starting new self-play game")
        env = XiangqiEnv()
        game_history = []
        move_count = 0
        moves = []  # Store moves for saving
        
        # Only show progress bar if not disabled
        pbar = None if self.disable_progress_bar else tqdm(desc="Self-play game", leave=False)
        vis = None
        if self.show_board:
            vis = XiangqiVisualizer(env)
            vis.draw_board()
        
        while not env.is_game_over:
            # Periodically force garbage collection
            if move_count % 10 == 0:
                gc.collect()
            
            # Clear MCTS tree every 10 moves to prevent memory buildup
            if move_count % 10 == 0:
                self.mcts.clear_tree()
            
            if vis:
                # Handle Pygame events more frequently
                for _ in range(10):  # Check events multiple times
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            vis.close()
                            return game_history
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                vis.close()
                                return game_history
                    pygame.time.wait(10)  # Small delay to keep UI responsive
            
            move_count += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Move {move_count}")
            
            # Get MCTS probabilities
            pi = self.mcts.search(env)
            
            # Store canonical state and MCTS probabilities
            state = env.get_canonical_state()
            state = np.array(state).reshape(14, 10, 9)  # Reshape for consistency
            game_history.append([
                state,
                pi,
                env.current_player
            ])
            
            # Select move (with temperature)
            valid_moves = env.get_valid_moves()
            valid_move_probs = []
            valid_move_indices = []
            
            for move in valid_moves:
                move_idx = self.mcts._move_to_index(move)
                valid_move_probs.append(pi[move_idx])
                valid_move_indices.append(move_idx)
            
            # Apply temperature
            if len(game_history) < 30:  # First 30 moves
                temperature = 1.2  # Higher temperature for more exploration
            else:
                temperature = 0.5  # Lower temperature for better moves
            
            probs = [p ** (1/temperature) for p in valid_move_probs]
            
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
                
                # Select move
                move_idx = np.random.choice(len(valid_moves), p=probs)
                action = valid_moves[move_idx]
                moves.append(action)  # Record the move
                
                if vis:
                    vis.animate_move(action[0], action[1])
                
                # Log the selected move
                logger.debug(f"Selected move {action} with probability {probs[move_idx]:.3f}")
                
                env.step(action)
                if vis:
                    vis.draw_board()
            
            # Force draw after too many moves
            if move_count >= self.max_moves:  # Use config max_moves
                logger.info("Game drawn due to move limit")
                self.save_game(moves, env.winner)
                return [(state, pi, 0) for state, pi, player in game_history]
        
        if pbar is not None:
            pbar.close()
        if vis:
            vis.close()
        
        # Save game to CSV
        value = env.get_reward()
        self.save_game(moves, env.winner)
        
        # Log game completion
        logger.info(f"Game completed after {move_count} moves. Result: {env.get_reward()}")
        
        # Convert game history to training data
        # Each state gets the final game outcome as its value target
        # value is +1 for win, -1 for loss from the perspective of the player who made the move
        return [(state, pi, value * (1 if player == env.current_player else -1)) 
                for state, pi, player in game_history]
        
    def save_game(self, moves, winner):
        """Save game to CSV file"""
        self.game_id += 1
        moves_str = " ".join(self._move_to_string(move) for move in moves)
        
        # Get result string based on winner
        if winner is None:  # Draw
            result_str = "和棋"
        else:  # Win/Loss
            result_str = "红方胜" if winner else "黑方胜"
            
        # Get current date
        from datetime import datetime
        current_date = datetime.now().strftime("%Y.%m.%d")
        
        # Create or append to CSV file
        csv_path = os.path.join(self.games_dir, "selfplay_games.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("game_id,moves,num_moves,event,date,red_player,black_player,result\n")
                
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f'{self.game_id},"{moves_str}",{len(moves)},"AlphaZero自我对弈",'
                   f'{current_date},"AI_Red","AI_Black",{result_str}\n')
            
    def _move_to_string(self, move):
        """Convert move tuple to string format like '9,6,7,4'"""
        (from_row, from_col), (to_row, to_col) = move
        return f"{from_row},{from_col},{to_row},{to_col}"
    