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

# Set start method to spawn
if __name__ == '__main__':
    mp.set_start_method('spawn')

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Training configuration and hyperparameters"""
    def __init__(self, num_iterations=1000, num_selfplay_games=100, batch_size=256, 
                 training_steps=1000, eval_interval=10, max_buffer_size=500000):
        self.num_iterations = num_iterations
        self.num_selfplay_games = num_selfplay_games
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.eval_interval = eval_interval
        self.max_buffer_size = max_buffer_size

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
        """Clear the search tree to free memory"""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()

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
            
        return move_probs
    
    def _simulate(self, env, state, move_count=0, position_history=None):
        """Simulate one instance of MCTS"""
        if position_history is None:
            position_history = set()
            
        # Check for repetition
        current_position = env.get_canonical_state()
        if current_position in position_history:
            return 0.0  # Draw due to repetition
        position_history.add(current_position)
        
        # Check termination conditions
        if move_count >= self.max_moves:
            # Return the value prediction from the neural network for this state
            state_array = np.array([env.get_state()])
            state_tensor = torch.FloatTensor(state_array).to(self.model.device)
            with torch.no_grad():
                _, value = self.model(state_tensor)
            return -value.item()
            
        if env.is_game_over:
            return -env.get_reward()
            
        if state not in self.Ps:
            # Leaf node - evaluate position
            state_array = np.array([env.get_state()])
            state_tensor = torch.FloatTensor(state_array).to(self.model.device)
            policy, value = self.model(state_tensor)
            self.Ps[state] = torch.softmax(policy[0], dim=0).cpu().detach().numpy()
            return -value.item()
            
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
        self.mcts = MCTS(model, num_simulations=num_sims, max_moves=200, disable_progress_bar=disable_progress_bar)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=config.max_buffer_size)
        self.writer = SummaryWriter("logs/tensorboard")
        
        # Create games directory if it doesn't exist
        self.games_dir = "logs/games"
        os.makedirs(self.games_dir, exist_ok=True)
        self.game_id = 0
        
    def _move_to_string(self, move):
        """Convert move tuple to string format like '9,6,7,4'"""
        (from_row, from_col), (to_row, to_col) = move
        return f"{from_row},{from_col},{to_row},{to_col}"
        
    def save_game(self, moves, result):
        """Save game to CSV file"""
        self.game_id += 1
        moves_str = " ".join(self._move_to_string(move) for move in moves)
        
        # Format result
        if result == 1:
            result_str = "红方胜"
        elif result == -1:
            result_str = "黑方胜"
        else:
            result_str = "和棋"
            
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
            
            # Store state and MCTS probabilities
            game_history.append([
                env.get_state(),
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
            if len(game_history) < 30:
                probs = [p ** (1/1.0) for p in valid_move_probs]
            else:
                probs = [p ** (1/0.5) for p in valid_move_probs]
            
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
        
        if pbar is not None:
            pbar.close()
        if vis:
            vis.close()
        
        # Save game to CSV
        value = env.get_reward()
        self.save_game(moves, value)
        
        return [(state, pi, value * player) for state, pi, player in game_history]
    
    def train(self):
        """Main training loop"""
        try:
            # Main training iterations
            for iteration in tqdm(range(self.config.num_iterations), desc="Training iterations"):
                # Self-play phase
                selfplay_pbar = tqdm(range(self.config.num_selfplay_games), 
                                   desc=f"Self-play games (iter {iteration})", 
                                   leave=False)
                
                for _ in selfplay_pbar:
                    game_history = self.self_play()
                    self.replay_buffer.extend(game_history)
                    # Update progress bar with buffer size
                    selfplay_pbar.set_postfix({
                        'buffer_size': len(self.replay_buffer),
                        'last_game_len': len(game_history)
                    })
                
                # Training phase
                if len(self.replay_buffer) >= self.config.batch_size:
                    train_pbar = tqdm(range(self.config.training_steps), 
                                    desc=f"Training steps (iter {iteration})", 
                                    leave=False)
                    
                    for step in train_pbar:
                        batch = random.sample(self.replay_buffer, self.config.batch_size)
                        policy_loss, value_loss = self.train_on_batch(batch)
                        
                        # Update progress bar with losses
                        train_pbar.set_postfix({
                            'policy_loss': f'{policy_loss:.3f}',
                            'value_loss': f'{value_loss:.3f}'
                        })
                        
                        # Log losses
                        self.writer.add_scalar('Loss/Policy', policy_loss, iteration * self.config.training_steps + step)
                        self.writer.add_scalar('Loss/Value', value_loss, iteration * self.config.training_steps + step)
                
                # Evaluation phase
                if iteration % self.config.eval_interval == 0:
                    self.evaluate()
                    
                # Save checkpoint
                if iteration % 100 == 0:
                    self.save_checkpoint(iteration)
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            self.save_checkpoint("interrupted")
        finally:
            self.writer.close()
    
    def train_on_batch(self, batch):
        """Train on a batch of data"""
        states, pis, values = zip(*batch)
        # Convert to numpy arrays first
        states = np.array(states)
        pis = np.array(pis)
        values = np.array(values)
        
        # Then convert to tensors
        states = torch.FloatTensor(states).to(self.model.device)
        pis = torch.FloatTensor(pis).to(self.model.device)
        values = torch.FloatTensor(values).to(self.model.device)
        
        # Get predictions
        policy_logits, value_preds = self.model(states)
        
        # Calculate losses
        policy_loss = -torch.mean(torch.sum(pis * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = F.mse_loss(value_preds.squeeze(), values)
        total_loss = policy_loss + value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
        return policy_loss.item(), value_loss.item()

    def evaluate(self):
        """Evaluate the current model strength"""
        # TODO: Implement evaluation against a fixed opponent or previous version
        pass
        
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
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

def self_play_worker(game_queue, model_state_dict, config):
    """Standalone worker function for parallel self-play"""
    try:
        # Create new model instance for this worker and force it to CPU
        model = XiangqiHybridNet()
        model.load_state_dict(model_state_dict)
        model = model.to('cpu')
        model.eval()  # Set to evaluation mode for self-play
        
        # Create a minimal trainer instance just for self-play
        trainer = AlphaZeroTrainer(model, config, show_board=False, disable_progress_bar=True)
        
        while True:
            try:
                with torch.no_grad():  # Disable gradient computation during self-play
                    game_history = trainer.self_play()
                game_queue.put(game_history)
            except Exception as e:
                logger.error(f"Error in self-play worker: {e}")
                break
    except Exception as e:
        logger.error(f"Worker initialization error: {e}")

class ParallelAlphaZeroTrainer(AlphaZeroTrainer):
    def __init__(self, model, config, num_workers=4, show_board=False):
        super().__init__(model, config, show_board)
        self.num_workers = num_workers

    def parallel_self_play(self):
        """Execute self-play games in parallel"""
        # Create a queue for collecting game results
        game_queue = mp.Queue()
        
        # Get model state dict and move it to CPU for workers
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        # Create worker processes
        workers = []
        logger.info(f"Starting {self.num_workers} worker processes...")
        
        for i in range(self.num_workers):
            p = mp.Process(
                target=self_play_worker,
                args=(game_queue, cpu_state_dict, self.config)
            )
            p.start()
            workers.append(p)
            
        # Collect results
        games_collected = 0
        all_games = []
        
        # Only show progress bar in main process
        pbar = tqdm(total=self.config.num_selfplay_games, 
                   desc="Collecting parallel self-play games",
                   position=0)
        
        while games_collected < self.config.num_selfplay_games:
            try:
                game_history = game_queue.get(timeout=300)  # 5 minute timeout
                all_games.extend(game_history)
                games_collected += 1
                pbar.update(1)
                pbar.set_postfix({'collected': games_collected})
                
                if games_collected >= self.config.num_selfplay_games:
                    break
                    
            except Exception as e:
                logger.error(f"Error collecting game results: {e}")
                break
                
        pbar.close()
        
        # Terminate workers
        logger.info("Terminating worker processes...")
        for w in workers:
            w.terminate()
            w.join()
            
        return all_games

    def train(self):
        """Main training loop with parallel self-play"""
        try:
            # Main training iterations
            for iteration in tqdm(range(self.config.num_iterations), desc="Training iterations"):
                # Parallel self-play phase
                game_histories = self.parallel_self_play()
                
                # Log the number of games collected
                logger.info(f"Collected {len(game_histories)} game positions")
                
                self.replay_buffer.extend(game_histories)
                logger.info(f"Replay buffer size: {len(self.replay_buffer)}")
                
                # Training phase
                if len(self.replay_buffer) >= self.config.batch_size:
                    train_pbar = tqdm(range(self.config.training_steps), 
                                    desc=f"Training steps (iter {iteration})", 
                                    leave=False)
                    
                    for step in train_pbar:
                        batch = random.sample(self.replay_buffer, self.config.batch_size)
                        policy_loss, value_loss = self.train_on_batch(batch)
                        
                        train_pbar.set_postfix({
                            'policy_loss': f'{policy_loss:.3f}',
                            'value_loss': f'{value_loss:.3f}'
                        })
                        
                        self.writer.add_scalar('Loss/Policy', policy_loss, 
                                             iteration * self.config.training_steps + step)
                        self.writer.add_scalar('Loss/Value', value_loss, 
                                             iteration * self.config.training_steps + step)
                
                # Evaluation and checkpoint saving
                if iteration % self.config.eval_interval == 0:
                    self.evaluate()
                    
                if iteration % 100 == 0:
                    self.save_checkpoint(iteration)
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            self.save_checkpoint("interrupted")
        finally:
            self.writer.close()

def test_mcts():
    """Test MCTS behavior"""
    model = XiangqiHybridNet()
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
    args = parser.parse_args()

    if args.test_mcts:
        test_mcts()
    else:
        config = TrainingConfig(
            num_iterations=args.iterations,
            num_selfplay_games=args.selfplay_games,
            batch_size=256,
            training_steps=1000,
            eval_interval=10
        )
        
        model = XiangqiHybridNet()
        
        if args.parallel:
            model.share_memory()  # Enable model sharing between processes
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
            
        trainer.train()