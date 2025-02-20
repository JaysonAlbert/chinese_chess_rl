import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import rpc
import redis
import pickle
import random
from collections import deque
from xiangqi_rl.train import AlphaZeroTrainer, TrainingConfig
from xiangqi_rl.model import XiangqiHybridNet
import json
from tqdm import tqdm
import time
import torch
import glob
import signal
from xiangqi_rl.agent import XiangqiAgent
from xiangqi_rl.environment import XiangqiEnv
from xiangqi_rl.logger import logger
from xiangqi_rl.mcts import MCTS

class DistributedAlphaZero:
    def __init__(
        self,
        rank,
        world_size,
        master_addr,
        master_port,
        redis_host,
        redis_port,
        config,
        is_master=False
    ):
        """
        Initialize distributed training setup
        
        Args:
            rank: ID of current node (0 for master, 1+ for workers)
            world_size: Total number of nodes
            master_addr: IP address of master node
            master_port: Port for master node
            redis_host: Redis server host
            redis_port: Redis server port
            config: Training configuration
            is_master: Whether this is the master node
            resume_from: Path to checkpoint to resume from, if any
            eval_interval: Override config's eval_interval if provided
        """
        self.rank = rank
        self.world_size = world_size
        self.is_master = is_master
        self.config = config
        
        # Initialize distributed backend
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank
        )
        
        # Initialize RPC for parameter server communication
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f'tcp://{master_addr}:{master_port}'
            )
        )
        
        self.redis_client = get_redis_client(redis_host, redis_port)
        
        # Initialize model
        self.model = XiangqiHybridNet().to(self.device)
        
        if self.is_master:
            self.start_iteration = 0  # Only master needs to track iteration
            # Master node maintains the official model and trainer
            self.trainer = AlphaZeroTrainer(
                self.model,
                config,
                show_board=False,
                disable_progress_bar=False
            )
            
            # Only master loads checkpoint
            if self.config.resume_from:
                self._load_checkpoint()
        
        # Wrap model with DDP
        if torch.cuda.is_available():
            self.model = DDP(self.model, device_ids=[self.rank])
        else:
            self.model = DDP(self.model)

        self.agent = XiangqiAgent(self.model.module, XiangqiEnv(), 
                                 num_simulations=self.config.num_simulations, 
                                 show_board=False,
                                 max_moves=self.config.max_moves,
                                 disable_progress_bar=False)
        
        # Create local replay buffer
        self.local_buffer = deque(maxlen=self.config.max_buffer_size)
        
        logger.info(f"Initialized node {self.rank + 1}/{self.world_size}")

        self.running = True
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.running = False

    def run(self):
        """Main training loop"""
        try:
            if self.is_master:
                self._run_master()
            else:
                self._run_worker()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in training loop: {e}", exc_info=True)
        finally:
            # Cleanup
            logger.info("Cleaning up distributed resources...")
            try:
                dist.destroy_process_group()
                rpc.shutdown()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)

    def _run_master(self):
        """Master node training loop"""
        logger.info("Starting master node")
        
        try:
            # Start from the loaded iteration number
            for iteration in range(self.start_iteration, self.config.num_iterations):
                if not self.running:
                    logger.info("Stopping master node gracefully...")
                    break
                    
                logger.info(f"Starting iteration {iteration}")
                
                # Clear completion flags at start of iteration
                self._clear_completion_flags()
                
                # Broadcast latest model to workers
                logger.info("Broadcasting model to workers...")
                self._broadcast_model()
                logger.info("Model broadcast complete")
                
                # Run self-play games locally
                logger.info("Starting local self-play games...")
                self._run_selfplay()
                logger.info(f"Completed local self-play with {len(self.local_buffer)} games")
                
                # Wait for workers to complete self-play
                logger.info("Waiting for workers to complete self-play...")
                self._wait_for_workers()
                logger.info("All workers completed self-play")
                
                # Aggregate replay buffers from all workers
                logger.info("Aggregating replay buffers from workers...")
                self._aggregate_replay_buffers()
                
                # Train on aggregated data
                logger.info("Starting training on aggregated data...")
                self._train_iteration()
                logger.info("Training iteration complete")
                
                # Save checkpoint before potential interruption
                if iteration % self.config.checkpoint_interval == 0:
                    logger.info(f"Saving checkpoint at iteration {iteration}...")
                    self.trainer.save_checkpoint(iteration)
                    logger.info("Checkpoint saved")
                
                # Replace trainer.evaluate() with distributed evaluation
                if iteration % self.config.eval_interval == 0:
                    logger.info("Starting distributed evaluation...")
                    self._run_distributed_evaluation()
                    logger.info("Distributed evaluation complete")
                
                # Clear Redis buffer periodically
                if iteration % 5 == 0:
                    logger.info("Clearing Redis buffer...")
                    self.redis_client.flushdb()
                    logger.info("Redis buffer cleared")

        except Exception as e:
            logger.error(f"Master node error: {e}", exc_info=True)
        finally:
            # Save final checkpoint
            if self.is_master:
                logger.info("Saving final checkpoint...")
                self.trainer.save_checkpoint("interrupted")

    def _run_worker(self):
        """Worker node training loop"""
        logger.info(f"Starting worker node {self.rank}")
        
        try:
            while self.running:
                    
                start_time = time.time()
                # Receive latest model from master
                logger.info(f"Worker {self.rank}: Waiting for model update...")
                self._receive_model()
                logger.info(f"Worker {self.rank}: Received updated model")

                 # Check if we're in evaluation mode
                eval_config_data = self.redis_client.get("evaluation_config")
                if eval_config_data:
                    eval_config = json.loads(eval_config_data)
                    if eval_config and eval_config['mode'] == 'evaluation':
                        self._run_evaluation_worker()
                        continue
                
                # Run self-play games
                logger.info(f"Worker {self.rank}: Starting self-play games...")
                self._run_selfplay()
                logger.info(f"Worker {self.rank}: Completed self-play with {len(self.local_buffer)} states from self-play")
                
                # Upload replay buffer to Redis
                logger.info(f"Worker {self.rank}: Uploading replay buffer to Redis...")
                self._upload_buffer()
                logger.info(f"Worker {self.rank}: Replay buffer uploaded")
                
                # Signal completion to master
                logger.info(f"Worker {self.rank}: Signaling completion...")
                self._signal_completion()
                logger.info(f"Worker {self.rank}: Waiting for next iteration")
                
                # Add timeout check
                if time.time() - start_time > self.config.iteration_timeout:
                    logger.warning(f"Worker {self.rank} iteration timed out")
                    continue
                
                # Wait for next iteration with shorter sleep
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Worker node {self.rank} error: {e}", exc_info=True)
        finally:
            logger.info(f"Worker node {self.rank} shutting down...")

    def _run_selfplay(self):
        """Run self-play games and add to local buffer"""
        total_games_needed = self.config.games_per_iteration
        games_completed_key = "games_completed"
        games_played = 0  # Track actual number of games played
        
        if self.is_master:
            # Master initializes the counter
            self.redis_client.set(games_completed_key, "0")
        
        while True:
            # Atomically increment games counter
            current_games = self.redis_client.incr(games_completed_key)
            
            if current_games > total_games_needed:
                logger.info(f"Node {self.rank}: Required number of games completed")
                break
            
            # Run one game
            logger.debug(f"Node {self.rank}: Starting game {current_games}/{total_games_needed}")
            game_history = self.agent.self_play()
            self.local_buffer.extend(game_history)
            games_played += 1  # Increment games played counter
            
            # Log progress periodically
            if current_games % 10 == 0:
                logger.info(f"Total games completed: {current_games}/{total_games_needed}")
        
        logger.info(f"Node {self.rank} completed {games_played} games")

    def _broadcast_model(self):
        """Broadcast model parameters from master to workers"""
        if self.is_master:
            logger.info("Master broadcasting model parameters...")
        else:
            logger.info(f"Worker {self.rank} waiting for model parameters...")
        
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        
        if self.is_master:
            logger.info("Model broadcast complete")
        else:
            logger.info(f"Worker {self.rank} received model parameters")

    def _receive_model(self):
        """Workers receive initial model parameters from master"""
        logger.info(f"Worker {self.rank} waiting for initial model parameters...")
        self._broadcast_model()  # Reuse broadcast logic
        logger.info(f"Worker {self.rank} received initial model parameters")

        # Update agent's model with the received parameters
        self.agent.model.load_state_dict(self.model.module.state_dict())

    def _upload_buffer(self):
        """Upload local replay buffer to Redis"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                buffer_key = f"buffer:{self.rank}"
                buffer_data = pickle.dumps(list(self.local_buffer))
                self.redis_client.set(buffer_key, buffer_data)
                # Clear local buffer after successful upload
                self.local_buffer.clear()
                return
            except (redis.RedisError, pickle.PickleError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to upload buffer after {max_retries} attempts: {e}", exc_info=True)
                    raise
                time.sleep(1)

    def _aggregate_replay_buffers(self):
        """Aggregate replay buffers from all workers and master"""
        new_data = []
        
        # Add master's local buffer first
        new_data.extend(list(self.local_buffer))
        logger.info(f"Added master's buffer with {len(self.local_buffer)} examples")
        # Clear master's local buffer after using it
        self.local_buffer.clear()
        
        # Collect data from all workers
        for worker_rank in range(1, self.world_size):  # Start from 1 to skip master
            buffer_key = f"buffer:{worker_rank}"
            buffer_data = self.redis_client.get(buffer_key)
            if buffer_data:
                try:
                    worker_buffer = pickle.loads(buffer_data)
                    new_data.extend(worker_buffer)
                    logger.info(f"Added worker {worker_rank}'s buffer with {len(worker_buffer)} examples")
                except (pickle.PickleError, EOFError) as e:
                    logger.error(f"Error unpickling buffer from worker {worker_rank}: {e}", exc_info=True)
                    continue
        
        # Combine existing buffer with new data
        all_data = list(self.trainer.replay_buffer) + new_data
        logger.info(f"Combined buffer size: {len(all_data)} (existing: {len(self.trainer.replay_buffer)}, new: {len(new_data)})")
        
        # Sample if total data exceeds max buffer size
        if len(all_data) > self.config.max_buffer_size:
            all_data = random.sample(all_data, self.config.max_buffer_size)
        
        # Update master's replay buffer
        self.trainer.replay_buffer = deque(all_data, maxlen=self.config.max_buffer_size)
        logger.info(f"Final aggregated buffer size: {len(self.trainer.replay_buffer)}")

    def _train_iteration(self):
        """Train on aggregated data"""
        buffer_size = len(self.trainer.replay_buffer)
        if buffer_size >= self.config.min_buffer_size:
            logger.info(f"Starting training with buffer size: {buffer_size}")
            progress_bar = tqdm(range(self.config.steps_per_iteration), desc="Training")
            for step in progress_bar:
                batch = random.sample(self.trainer.replay_buffer, self.config.batch_size)
                loss = self.trainer.train_on_batch(batch)
                
                # Handle loss being a tuple or single value
                if isinstance(loss, tuple):
                    total_loss = sum(loss)  # Sum all loss components
                    loss_str = f"Total: {total_loss:.4f} ("
                    loss_str += ", ".join(f"{l:.4f}" for l in loss)
                    loss_str += ")"
                else:
                    loss_str = f"{loss:.4f}"
                
                if step % 100 == 0:  # Log every 100 steps
                    progress_bar.set_postfix({"Loss": loss_str})
                    logger.info(f"Training step {step}/{self.config.steps_per_iteration}, Loss: {loss_str}")
        else:
            logger.info(f"Skipping training - insufficient data in buffer ({buffer_size} < {self.config.min_buffer_size})")

    def _signal_completion(self):
        """Signal completion of current iteration"""
        self.redis_client.set(f"complete:{self.rank}", "1")

    def _wait_for_workers(self):
        """Wait for all workers to complete current iteration"""
        max_retries = 120
        retry_count = 0
        while retry_count < max_retries:
            all_complete = True
            for worker_rank in range(1, self.world_size):
                if not self.redis_client.get(f"complete:{worker_rank}"):
                    all_complete = False
                    break
            if all_complete:
                # Clear completion flags after confirming all workers are done
                for worker_rank in range(1, self.world_size):
                    self.redis_client.delete(f"complete:{worker_rank}")
                return
            time.sleep(3)
            retry_count += 1
        
        logger.error("Timeout waiting for workers to complete", exc_info=True)
        raise TimeoutError("Workers did not complete in time")

    def _clear_completion_flags(self):
        """Clear completion flags and game counter for all workers"""
        for worker_rank in range(1, self.world_size):
            self.redis_client.delete(f"complete:{worker_rank}")
        # Also clear the games counter
        self.redis_client.delete("games_completed")

    def _load_checkpoint(self):
        """Load checkpoint and restore all saved states"""
        if not self.config.resume_from or not os.path.exists(self.config.resume_from):
            return

        logger.info(f"Loading checkpoint from {self.config.resume_from}")
        try:
            checkpoint = torch.load(self.config.resume_from, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set starting iteration
            self.start_iteration = checkpoint['iteration'] + 1
            logger.info(f"Resuming from iteration {self.start_iteration}")
            
            # For master node, restore additional training state
            if self.is_master:
                # Load optimizer state
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Restore replay buffer if it exists
                if 'replay_buffer' in checkpoint:
                    self.trainer.replay_buffer = deque(checkpoint['replay_buffer'], 
                                                    maxlen=self.config.max_buffer_size)
                    logger.info(f"Restored replay buffer with {len(self.trainer.replay_buffer)} examples")
                
                # Restore global step counter
                if 'global_step' in checkpoint:
                    self.trainer.global_step = checkpoint['global_step']
                    
                logger.info(f"Successfully restored full training state from checkpoint")
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            raise

    def _run_distributed_evaluation(self):
        """Run evaluation games distributed across workers"""
        if not self.is_master:
            return
        
        logger.info("Starting distributed evaluation")
        
        # Signal workers to start evaluation mode and set total games needed
        eval_config = {
            'mode': 'evaluation',
            'total_games': self.config.eval_games * 2,  # Each game played twice (switching sides)
            'games_completed': 0
        }
        self.redis_client.set("evaluation_config", json.dumps(eval_config))
        
        wins = 0
        draws = 0
        total_games = 0
        
        try:
            # Wait for workers to complete evaluation games
            for worker_rank in range(1, self.world_size):
                eval_key = f"eval_results:{worker_rank}"
                # Wait with timeout
                for _ in range(360):  # 360 second timeout
                    result = self.redis_client.get(eval_key)
                    if result:
                        worker_results = json.loads(result)
                        wins += worker_results['wins']
                        draws += worker_results['draws']
                        total_games += worker_results['total']
                        break
                    time.sleep(1)
            
            if total_games > 0:
                win_rate = (wins + draws) / total_games
                logger.info(f"Evaluation complete - Win rate against best model: {win_rate:.2%}")
                
                # Save new best model if significantly better
                if win_rate > 0.55:  # 55% win rate threshold
                    logger.info("New model performs better - saving as best model")
                    torch.save(self.model.module.state_dict(), self.trainer.best_model_path)
                    if self.trainer.best_model:
                        self.trainer.best_model.load_state_dict(self.model.module.state_dict())
                    
        finally:
            # Clear evaluation mode
            self.redis_client.delete("evaluation_config")
            # Clear results
            for worker_rank in range(1, self.world_size):
                self.redis_client.delete(f"eval_results:{worker_rank}")

    def _run_evaluation_worker(self):
        """Run evaluation games on worker node"""
        logger.info(f"Worker {self.rank} starting evaluation")
        
        # Create evaluation MCTS with appropriate device
        eval_mcts = MCTS(self.model.module, num_simulations=400,
                         max_moves=self.config.max_moves,
                         disable_progress_bar=True)
                         
        # Load best model for comparison
        best_model = XiangqiHybridNet().to(self.device)
        best_model_state = torch.load(self.trainer.best_model_path, 
                                     map_location=self.device)
        best_model.load_state_dict(best_model_state)
        
        best_mcts = MCTS(best_model, num_simulations=400,
                         max_moves=self.config.max_moves, 
                         disable_progress_bar=True)
        
        wins = 0
        draws = 0
        total_games = 0
        
        while True:
            # Get current evaluation config
            eval_config_data = self.redis_client.get("evaluation_config")
            if not eval_config_data:
                break
            eval_config = json.loads(eval_config_data)
            if eval_config['mode'] != 'evaluation':
                break
            
            # Check if we've reached the total games needed
            current_games = self.redis_client.incr("evaluation_games_completed", 2)
            if current_games > eval_config['total_games']:
                break
            
            # Play one game as red
            result = self.play_evaluation_game(eval_mcts, best_mcts)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 0.5
            total_games += 1
            
            # Play one game as black
            result = self.play_evaluation_game(best_mcts, eval_mcts)
            if result == -1:  # Win for current model as black
                wins += 1
            elif result == 0:
                draws += 0.5
            total_games += 1
        
        # Save results to Redis
        results = {
            'wins': wins,
            'draws': draws,
            'total': total_games
        }
        self.redis_client.set(f"eval_results:{self.rank}", json.dumps(results))
        logger.info(f"Worker {self.rank} completed evaluation: {results}")

def find_latest_checkpoint():
    """Find the latest checkpoint file in the checkpoints directory"""
    checkpoint_files = glob.glob('logs/checkpoints/model_iteration_*.pt')
    if not checkpoint_files:
        return None
        
    # Sort by iteration number, ignoring special cases like 'interrupted'
    def get_iteration_num(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            return -1  # Return -1 for special cases like 'interrupted'
            
    # Sort by iteration number in filename
    latest_checkpoint = max(checkpoint_files, key=get_iteration_num)
    return latest_checkpoint

def get_redis_client(redis_host, redis_port):
    """Create and test Redis connection"""
    try:
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        redis_client.ping()
        return redis_client
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        raise

def get_rank_from_redis(redis_client, world_size, clear_rank=False):
    """Get available rank from Redis using atomic operations"""
    rank_key = "xiangqi_training:rank_counter"
    
    # Clear rank counter if requested
    if clear_rank:
        redis_client.delete(rank_key)
        redis_client.delete("evaluation_config")
        logger.info("Cleared rank counter in Redis")
    
    try:
        # Atomically increment and get rank
        rank = redis_client.incr(rank_key)
        rank = rank - 1  # Convert to 0-based index
        
        if rank >= world_size:
            # Reset counter if we exceeded world_size
            redis_client.set(rank_key, "0")
            raise RuntimeError(f"No ranks available. Got rank {rank} but world_size is {world_size}")
            
        return rank
    except redis.RedisError as e:
        logger.error(f"Failed to get rank from Redis: {e}", exc_info=True)
        raise

def launch_distributed_training(
    world_size,
    master_addr,
    master_port,
    redis_host,
    redis_port,
    config,
    clear_rank=False
):
    """Launch function for distributed training"""
    # Create Redis client that will be reused
    redis_client = get_redis_client(redis_host, redis_port)
    
    # Get rank from Redis
    rank = get_rank_from_redis(redis_client, world_size, clear_rank)
    logger.info(f"Obtained rank {rank} from Redis")

    is_master=(rank == 0)
    # Find latest checkpoint if config.resume_from is not specified
    if is_master and config.resume_from is None:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Found latest checkpoint: {latest_checkpoint}")
            config.resume_from = latest_checkpoint
        else:
            logger.info("No checkpoint found, starting from scratch")

    trainer = DistributedAlphaZero(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        redis_host=redis_host,
        redis_port=redis_port,
        config=config,
        is_master=is_master
    )
    trainer.run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--master-addr', type=str, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--redis-host', type=str, required=True)
    parser.add_argument('--redis-port', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--clear-rank', action='store_true',
                       help='Clear rank counter in Redis before starting')
    
    args = parser.parse_args()
    
    # Load config from JSON file
    with open(args.config) as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)
    
    launch_distributed_training(
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        config=config,
        clear_rank=args.clear_rank
    ) 