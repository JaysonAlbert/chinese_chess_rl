import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
from xiangqi_rl.environment import XiangqiEnv
from xiangqi_rl.model import XiangqiHybridNet
import os
from torch.utils.tensorboard import SummaryWriter
import random
import torch.multiprocessing as mp
from collections import deque
import math
import pygame
from xiangqi_rl.visualize import XiangqiVisualizer
import gc
from xiangqi_rl.logger import logger
import glob
import re
import psutil

# Set start method to spawn
if __name__ == '__main__':
    mp.set_start_method('spawn')


class TrainingConfig:
    """Training configuration and hyperparameters"""
    def __init__(self, 
                 num_iterations=1000,
                 games_per_iteration=100,    # Games to play per iteration
                 min_buffer_size=5000,      # Min games before training starts
                 batch_size=256,
                 steps_per_iteration=1000,   # Training steps per iteration
                 eval_interval=10, 
                 max_buffer_size=500000,
                 max_moves=200):            # Maximum moves per game
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.steps_per_iteration = steps_per_iteration
        self.eval_interval = eval_interval
        self.max_buffer_size = max_buffer_size
        self.max_moves = max_moves

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
            policy = torch.softmax(policy[0], dim=0).cpu().detach().numpy()
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

class AlphaZeroTrainer:
    def __init__(self, model, config, show_board=False, disable_progress_bar=False):
        self.model = model
        self.config = config
        self.show_board = show_board
        self.disable_progress_bar = disable_progress_bar
        num_sims = 50 if show_board else 200
        self.mcts = MCTS(model, num_simulations=num_sims, 
                        max_moves=config.max_moves,  
                        disable_progress_bar=disable_progress_bar)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=config.max_buffer_size)
        self.writer = SummaryWriter("logs/tensorboard")
        
        # Add global step counter for tensorboard
        self.global_step = 0
        
        # Create games directory if it doesn't exist
        self.games_dir = "logs/games"
        os.makedirs(self.games_dir, exist_ok=True)
        self.game_id = 0
        
        # Add path for best model
        self.best_model_path = "logs/checkpoints/best_model.pt"
        self.best_model = None
        self.load_best_model()  # Load best model if exists
        
        # Add checkpoint tracking
        self.start_iteration = 0
        self.checkpoint_dir = "logs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _move_to_string(self, move):
        """Convert move tuple to string format like '9,6,7,4'"""
        (from_row, from_col), (to_row, to_col) = move
        return f"{from_row},{from_col},{to_row},{to_col}"
        
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
            if move_count >= self.config.max_moves:  # Use config max_moves
                logger.info("Game drawn due to move limit")
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
    
    def train(self):
        """Main training loop more similar to AlphaGo Zero"""
        try:
            # Start from the last saved iteration
            for iteration in tqdm(range(self.start_iteration, self.config.num_iterations)):
                # Self-play phase - collect games_per_iteration new games
                selfplay_pbar = tqdm(range(self.config.games_per_iteration), 
                                   desc=f"Self-play games (iter {iteration})", 
                                   leave=False)
                
                for _ in selfplay_pbar:
                    game_history = self.self_play()
                    self.replay_buffer.extend(game_history)
                    selfplay_pbar.set_postfix({
                        'buffer_size': len(self.replay_buffer)
                    })

                    # Start training once we have enough data
                    if len(self.replay_buffer) >= self.config.min_buffer_size:
                        # Do some training steps after each game
                        train_steps = self.config.steps_per_iteration // self.config.games_per_iteration
                        train_pbar = tqdm(range(train_steps), 
                                        desc="Training steps", 
                                        leave=False)
                        
                        for step in train_pbar:
                            batch = random.sample(self.replay_buffer, self.config.batch_size)
                            policy_loss, value_loss = self.train_on_batch(batch)
                            
                            # Track losses
                            self.policy_losses.append(policy_loss)
                            self.value_losses.append(value_loss)
                            
                            train_pbar.set_postfix({
                                'p_loss': f'{policy_loss:.3f}',
                                'v_loss': f'{value_loss:.3f}'
                            })
                            
                            # Log average losses every log_interval steps
                            step_idx = iteration * self.config.steps_per_iteration + step
                            if step_idx % self.log_interval == 0 and self.policy_losses:
                                avg_policy_loss = sum(self.policy_losses) / len(self.policy_losses)
                                avg_value_loss = sum(self.value_losses) / len(self.value_losses)
                                self.writer.add_scalar('Loss/Policy', avg_policy_loss, step_idx)
                                self.writer.add_scalar('Loss/Value', avg_value_loss, step_idx)
                                # Clear loss tracking lists
                                self.policy_losses = []
                                self.value_losses = []
                
                # Evaluation phase
                if iteration % self.config.eval_interval == 0:
                    self.evaluate()
                    
                # Save checkpoint
                if iteration % 1 == 0:
                    self.save_checkpoint(iteration)
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            self.save_checkpoint("interrupted")
        finally:
            self.writer.close()
    
    def train_on_batch(self, batch):
        """Train on a batch of data"""
        batch_size = len(batch)
        logger.debug(f"Training on batch of size {batch_size}")
        
        states, pis, values = zip(*batch)
        states = np.array(states)
        pis = np.array(pis)
        values = np.array(values)
        
        states = torch.FloatTensor(states).to(self.model.device)
        pis = torch.FloatTensor(pis).to(self.model.device)
        values = torch.FloatTensor(values).to(self.model.device)
        
        policy_logits, value_preds = self.model(states)
        
        # Add temperature to policy
        temperature = 1.0
        policy_logits = policy_logits / temperature
        
        policy_loss = -torch.mean(torch.sum(pis * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = F.mse_loss(value_preds.squeeze(), values)
        
        # Add L2 regularization
        l2_lambda = 1e-4
        l2_reg = torch.tensor(0.).to(self.model.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        
        total_loss = policy_loss + value_loss + l2_lambda * l2_reg
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
        # Log losses directly to tensorboard using global step
        self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.global_step)
        self.writer.add_scalar('Loss/Value', value_loss.item(), self.global_step)
        self.global_step += 1
        
        return policy_loss.item(), value_loss.item()

    def load_best_model(self):
        """Load the best model if it exists"""
        if os.path.exists(self.best_model_path):
            self.best_model = XiangqiHybridNet(device=self.model.device)
            self.best_model.load_state_dict(torch.load(self.best_model_path))
            logger.info("Loaded best model from previous training")

    def evaluate(self):
        """Evaluate current model against best model"""
        if self.best_model is None:
            # If no best model exists, save current model as best
            torch.save(self.model.state_dict(), self.best_model_path)
            self.best_model = XiangqiHybridNet(device=self.model.device)
            self.best_model.load_state_dict(self.model.state_dict())
            return

        wins = 0
        draws = 0
        num_games = 10  # Number of evaluation games
        
        for game in tqdm(range(num_games), desc="Evaluation games"):
            # Play one game with current model as red, best model as black
            result = self.play_evaluation_game(self.model, self.best_model)
            if result == 1:  # Win for current model
                wins += 1
            elif result == 0:  # Draw
                draws += 0.5

            # Play one game with roles reversed
            result = self.play_evaluation_game(self.best_model, self.model)
            if result == -1:  # Win for current model (as black)
                wins += 1
            elif result == 0:  # Draw
                draws += 0.5

        win_rate = (wins + draws) / (num_games * 2)
        logger.info(f"Evaluation complete - Win rate against best model: {win_rate:.2%}")

        # If new model is significantly better, save it as best model
        if win_rate > 0.55:  # 55% win rate threshold
            torch.save(self.model.state_dict(), self.best_model_path)
            self.best_model.load_state_dict(self.model.state_dict())
            logger.info("New best model saved!")

    def play_evaluation_game(self, red_player, black_player):
        """Play a single evaluation game between two models"""
        env = XiangqiEnv()
        move_count = 0
        
        while not env.is_game_over and move_count < self.config.max_moves:
            current_model = red_player if env.current_player else black_player
            # Use lower number of MCTS simulations for faster evaluation
            mcts = MCTS(current_model, num_simulations=400, 
                       max_moves=self.config.max_moves,
                       disable_progress_bar=True)
            
            pi = mcts.search(env)
            valid_moves = env.get_valid_moves()
            
            # Select best move (no exploration during evaluation)
            best_move = None
            best_prob = -1
            
            for move in valid_moves:
                move_idx = mcts._move_to_index(move)
                if pi[move_idx] > best_prob:
                    best_prob = pi[move_idx]
                    best_move = move
            
            env.step(best_move)
            move_count += 1

        if move_count >= self.config.max_moves:
            return 0  # Draw
        return env.get_reward()

    def save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_dir = "logs/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_iteration_{iteration}.pt")
        
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer),
            'global_step': self.global_step,  # Save global step
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """Load the latest checkpoint or a specific checkpoint"""
        if checkpoint_path is None:
            # Find the latest checkpoint
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "model_iteration_*.pt"))
            if not checkpoints:
                logger.info("No checkpoints found, starting fresh training")
                return False
                
            checkpoint_path = max(checkpoints, key=lambda x: int(re.search(r'iteration_(\d+)', x).group(1)))
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_iteration = checkpoint['iteration'] + 1
            
            # Restore replay buffer if it exists
            if 'replay_buffer' in checkpoint:
                self.replay_buffer = deque(checkpoint['replay_buffer'], maxlen=self.config.max_buffer_size)
                logger.info(f"Restored replay buffer with {len(self.replay_buffer)} examples")
            
            # Restore global step counter
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            
            logger.info(f"Resuming training from iteration {self.start_iteration}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

def self_play_worker(game_queue, model_state_dict, config, disable_progress_bar=True):
    """Standalone worker function for parallel self-play"""
    try:
        # Create new model instance for this worker and explicitly configure for CPU
        model = XiangqiHybridNet(device='cpu')
        model.load_state_dict(model_state_dict)
        model.eval()  # Set to evaluation mode for self-play
        
        # Force model to CPU mode and optimize for CPU operations
        model = model.cpu()
        torch.set_num_threads(1)  # Restrict to single thread per process
        torch.set_num_interop_threads(1)
        
        # Disable gradient computation permanently for this worker
        torch.set_grad_enabled(False)
        
        # Create a minimal trainer instance just for self-play
        trainer = AlphaZeroTrainer(model, config, show_board=False, 
                                 disable_progress_bar=disable_progress_bar)
        
        while True:
            try:
                game_history = trainer.self_play()
                game_queue.put(game_history)
            except Exception as e:
                logger.error(f"Error in self-play worker: {e}")
                break
    except Exception as e:
        logger.error(f"Worker initialization error: {e}")

class ParallelAlphaZeroTrainer(AlphaZeroTrainer):
    def __init__(self, model, config, num_workers=4, show_board=False, disable_progress_bar=True):
        super().__init__(model, config, show_board, disable_progress_bar)
        self.num_workers = num_workers
        self.disable_progress_bar = disable_progress_bar
        
        # Set global PyTorch settings for the main process
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def parallel_self_play(self):
        """Execute self-play games in parallel"""
        logger.info(f"Starting parallel self-play with {self.num_workers} workers")
        
        # Create a queue for collecting game results
        game_queue = mp.Queue()
        
        # Get model state dict and ensure it's on CPU with contiguous memory
        cpu_state_dict = {k: v.cpu().contiguous() for k, v in self.model.state_dict().items()}
        
        # Create worker processes with explicit CPU affinity if possible
        workers = []
        logger.info(f"Starting {self.num_workers} worker processes...")
        
        try:
            for i in range(self.num_workers):
                p = mp.Process(
                    target=self_play_worker,
                    args=(game_queue, cpu_state_dict, self.config, True if i != 0 else False)
                )
                p.start()
                
                # Set CPU affinity for each worker process
                worker_process = psutil.Process(p.pid)
                # Assign each worker to a specific CPU core
                worker_process.cpu_affinity([i % psutil.cpu_count()])
                
                workers.append(p)
        except ImportError:
            # Fall back to normal process creation if psutil is not available
            for i in range(self.num_workers):
                p = mp.Process(
                    target=self_play_worker,
                    args=(game_queue, cpu_state_dict, self.config, True if i != 0 else False)
                )
                p.start()
                workers.append(p)

        # Collect results
        games_collected = 0
        all_games = []
        
        pbar = tqdm(total=self.config.games_per_iteration, 
                    desc="Collecting parallel self-play games",
                    position=0)
        
        try:
            while games_collected < self.config.games_per_iteration:
                try:
                    game_history = game_queue.get(timeout=3600)  # 5 minute timeout
                    all_games.extend(game_history)
                    games_collected += 1
                    pbar.update(1)
                    
                    # Add games to replay buffer immediately instead of accumulating
                    self.replay_buffer.extend(game_history)
                    current_buffer_size = len(self.replay_buffer)
                    
                    logger.info(f"Collected game {games_collected}/{self.config.games_per_iteration}. "
                              f"Buffer size: {current_buffer_size}")
                    
                    # Log memory usage periodically
                    if games_collected % 10 == 0:
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        logger.info(f"Memory usage - RSS: {memory_info.rss / 1024 / 1024:.1f}MB, "
                                  f"VMS: {memory_info.vms / 1024 / 1024:.1f}MB")
                    
                    # Start training if we have enough data
                    if current_buffer_size >= self.config.min_buffer_size:
                        # Do some training steps
                        train_steps = self.config.steps_per_iteration // self.config.games_per_iteration
                        train_pbar = tqdm(range(train_steps), 
                                        desc="Training steps", 
                                        leave=False)
                        
                        for step in train_pbar:
                            batch = random.sample(self.replay_buffer, min(self.config.batch_size, current_buffer_size))
                            policy_loss, value_loss = self.train_on_batch(batch)
                            
                            train_pbar.set_postfix({
                                'p_loss': f'{policy_loss:.3f}',
                                'v_loss': f'{value_loss:.3f}'
                            })

                    if games_collected >= self.config.games_per_iteration:
                        break
                        
                except Exception as e:
                    logger.error(f"Error collecting game results: {e}", exc_info=True)
                    break

        finally:
            pbar.close()
            
            # Terminate workers
            logger.info("Terminating worker processes...")
            for w in workers:
                w.terminate()
                w.join()
        
            # Log cleanup
            logger.info("Cleaning up parallel self-play resources")
        
        return len(self.replay_buffer)

    def train(self):
        """Main training loop with parallel self-play"""
        logger.info("Starting parallel training")
        try:
            # Start from the last saved iteration
            for iteration in tqdm(range(self.start_iteration, self.config.num_iterations)):
                logger.info(f"\nStarting iteration {iteration}/{self.config.num_iterations}")
                
                # Log GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    logger.info(f"GPU Memory allocated: {gpu_memory:.1f}MB")
                
                buffer_size = self.parallel_self_play()
                logger.info(f"Iteration {iteration} completed. Final buffer size: {buffer_size}")
                
                if iteration % self.config.eval_interval == 0:
                    logger.info(f"Running evaluation at iteration {iteration}")
                    self.evaluate()
                
                logger.info(f"Saving checkpoint at iteration {iteration}")
                self.save_checkpoint(iteration)
                    
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.save_checkpoint("interrupted")
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
        finally:
            logger.info("Training completed")
            self.writer.close()

def test_mcts():
    """Test MCTS behavior"""
    model = XiangqiHybridNet(device='cpu')
    mcts = MCTS(model, num_simulations=200)
    env = XiangqiEnv()
    
    # Test initial position
    print("Testing initial position...")
    pi = mcts.search(env, debug=True)
    
    # Test after a few moves
    print("\nTesting after some moves...")
    moves = [
        ((6, 4), (5, 4)),  # Red pawn forward
        ((3, 4), (4, 4)),  # Black pawn forward
        ((7, 1), (5, 2)),  # Red cannon to attack
    ]
    for move in moves:
        env.step(move)
        print(f"\nAfter move {move}:")
        pi = mcts.search(env, debug=True)
    
    # Verify properties
    print("\nVerifying MCTS properties:")
    print(f"1. Total simulations: {sum(mcts.Nsa.values())}")
    print(f"2. Number of explored states: {len(mcts.Ns)}")
    print(f"3. Policy sum close to 1: {pi.sum():.6f}")
    print(f"4. Max policy value: {pi.max():.6f}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    import multiprocessing as mp
    
    # Get CPU count and set default workers to 75% of available cores
    default_workers = max(1, int(mp.cpu_count() * 0.75))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--selfplay-games', type=int, default=100)
    parser.add_argument('--show-board', action='store_true', help='Show the game board visualization')
    parser.add_argument('--test-mcts', action='store_true', help='Run MCTS tests')
    parser.add_argument('--parallel', action='store_true', help='Use parallel training')
    parser.add_argument('--num-workers', type=int, default=default_workers, 
                       help=f'Number of parallel self-play workers (default: {default_workers})')
    parser.add_argument('--max-moves', type=int, default=200,
                       help='Maximum number of moves per game before forcing a draw')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str,
                       help='Resume training from specific checkpoint file')
    args = parser.parse_args()

    if args.test_mcts:
        test_mcts()
    else:
        config = TrainingConfig(
            num_iterations=args.iterations,
            games_per_iteration=args.selfplay_games,
            min_buffer_size=10000,
            batch_size=256,
            steps_per_iteration=1000,
            eval_interval=10,
            max_moves=args.max_moves
        )
        
        model = XiangqiHybridNet()
        
        if args.parallel:
            model.share_memory()
            trainer = ParallelAlphaZeroTrainer(
                model, 
                config, 
                num_workers=args.num_workers,
                show_board=args.show_board
            )
        else:
            trainer = AlphaZeroTrainer(
                model,
                config,
                show_board=args.show_board
            )

        # Load checkpoint if requested
        if args.resume or args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            
        trainer.train()