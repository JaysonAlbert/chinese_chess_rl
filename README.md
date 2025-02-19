# XiangQi Reinforcement Learning

This project implements a Chinese Chess (XiangQi) AI using deep reinforcement learning techniques inspired by AlphaGo Zero. The system learns to play XiangQi through self-play without any human knowledge input.

## Key Features

- Distributed training architecture supporting multiple workers
- Self-play based learning using Monte Carlo Tree Search (MCTS)
- Deep neural network combining policy and value networks
- Redis-based coordination for distributed training
- Checkpoint saving and loading for training resumption
- Configurable training parameters

## Architecture

The training system consists of:

- A hybrid neural network that predicts both move probabilities and position evaluation
- Distributed workers that generate self-play games using MCTS
- A parameter server that aggregates experiences and updates the model
- Redis for worker coordination and rank assignment
- Distributed data parallel (DDP) training across multiple GPUs

## Requirements

- Python 3.7+
- PyTorch
- Redis
- Additional dependencies in requirements.txt

## Usage

1. Start Redis server:
   ```bash
   redis-server
   ```

2. install the project
   ```
   pip install -e .
   ```


3. Configure training parameters in config.json:
   ```json
   {
     "num_iterations": 1000,
     "num_games_per_iteration": 100,
     "max_buffer_size": 100000,
     "batch_size": 256,
     "num_epochs": 10,
     "checkpoint_interval": 10
   }
   ```
4. Launch distributed training (start world-size number of processes):
   ```bash
   # Launch 4 processes for world-size=4
   python -m xiangqi_rl.distributed_trainer \
     --world-size 4 \
     --master-addr localhost \
     --master-port 29500 \
     --redis-host localhost \
     --redis-port 6379 \
     --config config.json &
   ```

5. Monitor training progress:
   ```bash
   tensorboard --logdir logs/
   ```
