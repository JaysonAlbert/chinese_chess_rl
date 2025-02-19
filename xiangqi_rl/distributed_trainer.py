import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import rpc
import redis
import pickle
import logging
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
                                 num_simulations=100, 
                                 show_board=False, 
                                 disable_progress_bar=False)
        
        # Create local replay buffer
        self.local_buffer = deque(maxlen=self.config.max_buffer_size)
        
        logger.info(f"Initialized node {self.rank}/{self.world_size}")

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
            logger.error(f"Error in training loop: {e}")
        finally:
            # Cleanup
            logger.info("Cleaning up distributed resources...")
            try:
                dist.destroy_process_group()
                rpc.shutdown()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

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
                
                # Evaluate and save checkpoints
                if iteration % self.config.eval_interval == 0:
                    logger.info("Starting evaluation...")
                    self.trainer.evaluate()
                    logger.info("Evaluation complete")
                
                # Clear Redis buffer periodically
                if iteration % 5 == 0:
                    logger.info("Clearing Redis buffer...")
                    self.redis_client.flushdb()
                    logger.info("Redis buffer cleared")

        except Exception as e:
            logger.error(f"Master node error: {e}")
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
                
                # Run self-play games
                logger.info(f"Worker {self.rank}: Starting self-play games...")
                self._run_selfplay()
                logger.info(f"Worker {self.rank}: Completed self-play with {len(self.local_buffer)} games")
                
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
            logger.error(f"Worker node {self.rank} error: {e}")
        finally:
            logger.info(f"Worker node {self.rank} shutting down...")

    def _run_selfplay(self):
        """Run self-play games and add to local buffer"""
        total_games_needed = self.config.games_per_iteration
        games_completed_key = "games_completed"
        
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
            
            # Log progress periodically
            if current_games % 10 == 0:
                logger.info(f"Total games completed: {current_games}/{total_games_needed}")
            
        games_played = len(self.local_buffer)
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
                    logger.error(f"Failed to upload buffer after {max_retries} attempts: {e}")
                    raise
                time.sleep(1)

    def _aggregate_replay_buffers(self):
        """Aggregate replay buffers from all workers and master"""
        all_data = []
        
        # Add master's local buffer first
        all_data.extend(list(self.local_buffer))
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
                    all_data.extend(worker_buffer)
                    logger.info(f"Added worker {worker_rank}'s buffer with {len(worker_buffer)} examples")
                except (pickle.PickleError, EOFError) as e:
                    logger.error(f"Error unpickling buffer from worker {worker_rank}: {e}")
                    continue
        
        # Sample if total data exceeds max buffer size
        if len(all_data) > self.config.max_buffer_size:
            all_data = random.sample(all_data, self.config.max_buffer_size)
        
        # Update master's replay buffer
        self.trainer.replay_buffer = deque(all_data, maxlen=self.config.max_buffer_size)
        logger.info(f"Aggregated buffer size: {len(self.trainer.replay_buffer)} from {len(all_data)} total examples")

    def _train_iteration(self):
        """Train on aggregated data"""
        buffer_size = len(self.trainer.replay_buffer)
        if buffer_size >= self.config.min_buffer_size:
            logger.info(f"Starting training with buffer size: {buffer_size}")
            for step in tqdm(range(self.config.steps_per_iteration), desc="Training steps"):
                batch = random.sample(self.trainer.replay_buffer, self.config.batch_size)
                loss = self.trainer.train_on_batch(batch)
                if step % 100 == 0:  # Log every 100 steps
                    logger.info(f"Training step {step}/{self.config.steps_per_iteration}, Loss: {loss:.4f}")
        else:
            logger.info(f"Skipping training - insufficient data in buffer ({buffer_size} < {self.config.min_buffer_size})")

    def _signal_completion(self):
        """Signal completion of current iteration"""
        self.redis_client.set(f"complete:{self.rank}", "1")

    def _wait_for_workers(self):
        """Wait for all workers to complete current iteration"""
        max_retries = 100
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
            time.sleep(1)
            retry_count += 1
        
        logger.error("Timeout waiting for workers to complete")
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
            logger.error(f"Error loading checkpoint: {e}")
            raise

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
        logger.error(f"Failed to connect to Redis: {e}")
        raise

def get_rank_from_redis(redis_client, world_size, clear_rank=False):
    """Get available rank from Redis using atomic operations"""
    rank_key = "xiangqi_training:rank_counter"
    
    # Clear rank counter if requested
    if clear_rank:
        redis_client.delete(rank_key)
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
        logger.error(f"Failed to get rank from Redis: {e}")
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