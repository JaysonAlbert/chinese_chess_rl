import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import rpc
import torch.distributed.rpc as rpc
from collections import deque
import redis
import pickle
import logging
import random
from xiangqi_rl.train import AlphaZeroTrainer, TrainingConfig
from xiangqi_rl.model import XiangqiHybridNet
import json
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

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
        """
        self.rank = rank
        self.world_size = world_size
        self.is_master = is_master
        self.config = config
        
        # Initialize distributed backend
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        
        # Initialize distributed process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL backend for GPU training
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
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        
        # Initialize model
        self.model = XiangqiHybridNet().cuda()
        self.model = DDP(self.model)
        
        if self.is_master:
            # Master node maintains the official model
            self.trainer = AlphaZeroTrainer(
                self.model.module,
                config,
                show_board=False,
                disable_progress_bar=False
            )
            
        # Create local replay buffer
        self.local_buffer = deque(maxlen=config.max_buffer_size)
        
        logger.info(f"Initialized node {rank}/{world_size}")

    def run(self):
        """Main training loop"""
        try:
            if self.is_master:
                self._run_master()
            else:
                self._run_worker()
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        finally:
            # Cleanup
            dist.destroy_process_group()
            rpc.shutdown()

    def _run_master(self):
        """Master node training loop"""
        logger.info("Starting master node")
        
        for iteration in range(self.config.num_iterations):
            logger.info(f"Starting iteration {iteration}")
            
            # Broadcast latest model to workers
            self._broadcast_model()
            
            # Run self-play games locally
            self._run_selfplay()
            
            # Wait for workers to complete self-play
            self._wait_for_workers()
            
            # Aggregate replay buffers from all workers
            self._aggregate_replay_buffers()
            
            # Train on aggregated data
            self._train_iteration()
            
            # Evaluate and save checkpoints
            if iteration % self.config.eval_interval == 0:
                self.trainer.evaluate()
                self.trainer.save_checkpoint(iteration)
                
            # Clear Redis buffer periodically
            if iteration % 5 == 0:
                self.redis_client.flushdb()

    def _run_worker(self):
        """Worker node training loop"""
        logger.info(f"Starting worker node {self.rank}")
        
        while True:
            # Receive latest model from master
            self._receive_model()
            
            # Run self-play games
            self._run_selfplay()
            
            # Upload replay buffer to Redis
            self._upload_buffer()
            
            # Signal completion to master
            self._signal_completion()
            
            # Wait for next iteration
            time.sleep(1)

    def _run_selfplay(self):
        """Run self-play games and add to local buffer"""
        num_games = self.config.games_per_iteration // self.world_size
        
        for _ in tqdm(range(num_games), desc=f"Self-play games (Node {self.rank})"):
            game_history = self.trainer.self_play()
            self.local_buffer.extend(game_history)

    def _broadcast_model(self):
        """Broadcast model parameters from master to workers"""
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def _receive_model(self):
        """Receive broadcasted model parameters on workers"""
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def _upload_buffer(self):
        """Upload local replay buffer to Redis"""
        buffer_key = f"buffer:{self.rank}"
        buffer_data = pickle.dumps(list(self.local_buffer))
        self.redis_client.set(buffer_key, buffer_data)

    def _aggregate_replay_buffers(self):
        """Aggregate replay buffers from all workers"""
        all_data = []
        
        # Collect data from all workers
        for worker_rank in range(self.world_size):
            buffer_key = f"buffer:{worker_rank}"
            buffer_data = self.redis_client.get(buffer_key)
            if buffer_data:
                worker_buffer = pickle.loads(buffer_data)
                all_data.extend(worker_buffer)
        
        # Update master's replay buffer
        self.trainer.replay_buffer = deque(all_data, maxlen=self.config.max_buffer_size)
        logger.info(f"Aggregated buffer size: {len(self.trainer.replay_buffer)}")

    def _train_iteration(self):
        """Train on aggregated data"""
        if len(self.trainer.replay_buffer) >= self.config.min_buffer_size:
            for _ in tqdm(range(self.config.steps_per_iteration), desc="Training steps"):
                batch = random.sample(self.trainer.replay_buffer, self.config.batch_size)
                self.trainer.train_on_batch(batch)

    def _signal_completion(self):
        """Signal completion of current iteration"""
        self.redis_client.set(f"complete:{self.rank}", "1")

    def _wait_for_workers(self):
        """Wait for all workers to complete current iteration"""
        for worker_rank in range(1, self.world_size):
            while not self.redis_client.get(f"complete:{worker_rank}"):
                time.sleep(0.1)
            self.redis_client.delete(f"complete:{worker_rank}")

def launch_distributed_training(
    rank,
    world_size,
    master_addr,
    master_port,
    redis_host,
    redis_port,
    config
):
    """Launch function for distributed training"""
    trainer = DistributedAlphaZero(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        redis_host=redis_host,
        redis_port=redis_port,
        config=config,
        is_master=(rank == 0)
    )
    trainer.run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--master-addr', type=str, required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--redis-host', type=str, required=True)
    parser.add_argument('--redis-port', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load config from JSON file
    with open(args.config) as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)
    
    launch_distributed_training(
        rank=args.rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        config=config
    ) 