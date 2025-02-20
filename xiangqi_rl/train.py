import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from xiangqi_rl.environment import XiangqiEnv
from xiangqi_rl.mcts import MCTS
from xiangqi_rl.model import XiangqiHybridNet
import os
from torch.utils.tensorboard import SummaryWriter
import random
import torch.multiprocessing as mp
from collections import deque
from xiangqi_rl.logger import logger
import glob
import re
import psutil
import queue

# Set start method to spawn
if __name__ == '__main__':
    mp.set_start_method('spawn')


class TrainingConfig:
    def __init__(
        self,
        num_iterations=1000,
        games_per_iteration=100,
        num_simulations=200,
        batch_size=128,
        steps_per_iteration=500,
        max_buffer_size=100000,
        min_buffer_size=10000,
        learning_rate=0.001,
        weight_decay=0.0001,
        dirichlet_alpha=0.3,
        exploration_constant=1.0,
        temperature=1.0,
        temperature_drop=10,
        eval_games=10,
        eval_interval=5,
        checkpoint_interval=5,
        max_moves=200,
        resume_from=None
    ):
        """
        Configuration for AlphaZero training
        
        Args:
            num_iterations: Total number of training iterations
            games_per_iteration: Number of self-play games per iteration
            num_simulations: Number of MCTS simulations per move
            batch_size: Training batch size
            steps_per_iteration: Number of training steps per iteration
            max_buffer_size: Maximum size of replay buffer
            min_buffer_size: Minimum samples needed before training starts
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization parameter
            dirichlet_alpha: Dirichlet noise parameter for root node
            exploration_constant: PUCT constant for MCTS
            temperature: Initial temperature for action selection
            temperature_drop: Move number when temperature drops to 0
            eval_games: Number of evaluation games to play
            eval_interval: How often to run evaluation (in iterations)
            checkpoint_interval: How often to save model checkpoints
            resume_from: Path to checkpoint to resume from
        """
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.steps_per_iteration = steps_per_iteration
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.temperature_drop = temperature_drop
        self.eval_games = eval_games
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.resume_from = resume_from
        self.max_moves = max_moves
        self.iteration_timeout = 3600  # 1 hour timeout for iterations

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
        # if win_rate > 0.55:  # 55% win rate threshold
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
                
            # More robust checkpoint file parsing
            iterations = []
            for checkpoint in checkpoints:
                match = re.search(r'iteration_(\d+)', checkpoint)
                if match:
                    iterations.append((int(match.group(1)), checkpoint))
            
            if not iterations:
                logger.info("No valid checkpoint files found, starting fresh training")
                return False
            
            # Get the checkpoint with highest iteration number
            _, checkpoint_path = max(iterations, key=lambda x: x[0])
        
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

def self_play_worker(game_queue, model_state_dict, config, device='cpu', disable_progress_bar=True):
    try:
        # Create new model instance for this worker with specified device
        model = XiangqiHybridNet(device=device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # Move model to specified device
        model = model.to(device)
        
        if device == 'cpu':
            # Optimize for CPU operations
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        else:
            # For GPU workers
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Set higher priority for GPU workers
            try:
                os.nice(-10)  # Give GPU workers higher priority
            except:
                pass
            # Pin memory for faster GPU transfers
            torch.cuda.set_device(int(device.split(':')[1]))
            
        # Disable gradient computation
        torch.set_grad_enabled(False)
        
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
    def __init__(self, model, config, num_workers=4,  show_board=False, disable_progress_bar=True):
        super().__init__(model, config, show_board, disable_progress_bar)
        self.num_workers = num_workers
        self.disable_progress_bar = disable_progress_bar

        # Set CUDA device for each GPU worker
        self.devices = []
        if torch.cuda.is_available():
            for i in range(self.num_workers):
                self.devices.append(f'cuda:{i % torch.cuda.device_count()}')
        else:
            self.devices = ['cpu'] * self.num_workers
            
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Initializing with {self.num_workers} workers")
        logger.info(f"Devices: {self.devices}")

    def parallel_self_play(self):
        """Execute self-play games in parallel using both CPU and GPU"""
        logger.info(f"Starting parallel self-play with {self.num_workers} workers")
        
        # Create a queue for collecting game results
        game_queue = mp.Queue()
        
        # Get model state dict and ensure it's on CPU with contiguous memory
        cpu_state_dict = {k: v.cpu().contiguous() for k, v in self.model.state_dict().items()}
        
        # Create worker processes
        workers = []
        logger.info("Starting worker processes...")
        
        try:
            
            # Start GPU or CPU workers
            for i in range(self.num_workers):
                device = self.devices[i]
                p = mp.Process(
                    target=self_play_worker,
                    args=(game_queue, cpu_state_dict, self.config, device, True if i != 0 else False)
                )
                p.start()
                workers.append(p)

        except Exception as e:
            logger.error(f"Error starting workers: {e}")
            for w in workers:
                w.terminate()
            raise e

        # Collect results
        games_collected = 0
        all_games = []
        
        pbar = tqdm(total=self.config.games_per_iteration, 
                    desc="Collecting parallel self-play games",
                    position=0)
        
        try:
            while games_collected < self.config.games_per_iteration:
                try:
                    # Add timeout to prevent infinite waiting
                    game_history = game_queue.get(timeout=3600)  # 1 hour timeout
                    all_games.extend(game_history)
                    games_collected += 1
                    pbar.update(1)
                    
                    # Add games to replay buffer immediately
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

                except queue.Empty:
                    logger.error("Timeout while waiting for games. Terminating workers and restarting collection.")
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
                w.join(timeout=5)  # Add timeout to prevent hanging
            
                # Force kill if process hasn't terminated
                if w.is_alive():
                    logger.warning(f"Force killing worker process {w.pid}")
                    try:
                        os.kill(w.pid, 9)  # SIGKILL
                    except:
                        pass
        
            # Clear the queue
            while not game_queue.empty():
                try:
                    game_queue.get_nowait()
                except:
                    pass
                
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