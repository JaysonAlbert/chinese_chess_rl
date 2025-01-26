import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pygame
import logging
import numpy as np
from environment import XiangqiEnv
from agent import XiangqiAgent
from model import XiangqiHybridNet
from visualize import XiangqiVisualizer
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import random
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Event
from collections import deque
import time
import multiprocessing
from tqdm.auto import tqdm
import queue

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pretrain_on_database(model, database_path, num_epochs=10, batch_size=64):
    """Pretrain the model on human games database"""
    logger.info("Loading game database...")
    games_df = pd.read_csv(database_path)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # 添加进度条
        progress_bar = tqdm(games_df.iterrows(), total=len(games_df), 
                          desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for _, game in progress_bar:
            env = XiangqiEnv()
            moves = game['moves'].split()
            
            # 批量处理来提高训练效率
            states = []
            target_moves = []
            
            for move_str in moves:
                try:
                    # 添加错误处理
                    from_pos = (int(move_str[1]), int(move_str[0]))
                    to_pos = (int(move_str[3]), int(move_str[2]))
                    
                    state = env._get_state()
                    states.append(state)
                    target_moves.append(env._move_to_index((from_pos, to_pos)))
                    
                    # 当收集够一个批次时进行训练
                    if len(states) == batch_size:
                        # Convert to numpy array first
                        states_array = np.array(states)
                        state_tensor = torch.FloatTensor(states_array)
                        target_moves_tensor = torch.LongTensor(target_moves)
                        
                        policy, value = model(state_tensor)
                        policy_loss = F.cross_entropy(policy, target_moves_tensor)
                        
                        optimizer.zero_grad()
                        policy_loss.backward()
                        optimizer.step()
                        
                        total_loss += policy_loss.item()
                        num_batches += 1
                        
                        # 清空批次
                        states = []
                        target_moves = []
                    
                    # 执行移动
                    env.step((from_pos, to_pos))
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid move: {move_str}, error: {e}")
                    continue
            
            # 处理剩余的数据
            if states:
                # Convert to numpy array first
                states_array = np.array(states)
                state_tensor = torch.FloatTensor(states_array)
                target_moves_tensor = torch.LongTensor(target_moves)
                
                policy, value = model(state_tensor)
                policy_loss = F.cross_entropy(policy, target_moves_tensor)
                
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
                
                total_loss += policy_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def play_games(rank, model, experience_queue, stop_event, num_episodes, max_moves, visualize=False, progress_queue=None):
    """Process function to play games and collect experience"""
    env = XiangqiEnv()
    agent = XiangqiAgent(model)
    
    if visualize and rank == 0:
        vis = XiangqiVisualizer(env)
    else:
        vis = None
    
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    try:
        episode = 0
        while not stop_event.is_set() and episode < num_episodes:
            state = env.reset()
            done = False
            episode_data = []
            move_count = 0
            
            while not done and move_count < max_moves:
                if vis:
                    vis.draw_board()
                    pygame.time.wait(100)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            stop_event.set()
                            break
                
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                    
                state_array = env._get_state()
                
                if random.random() < epsilon:
                    action = random.choice(valid_moves)
                else:
                    action = agent.select_action(state_array, valid_moves, temperature=1.0)
                
                next_state, reward, done = env.step(action)
                episode_data.append((state_array, action, reward))
                state = next_state
                move_count += 1
            
            experience_queue.put(episode_data)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode += 1
            
            # Report progress
            if progress_queue is not None:
                progress_queue.put(1)
            
    finally:
        if vis:
            vis.close()

def train_model(model, experience_queue, stop_event, batch_size=64, save_interval=1000):
    """Process function to train the model using collected experience"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    log_dir = "logs/tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    checkpoint_dir = "logs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    replay_buffer = deque(maxlen=100000)
    running_reward = 0
    episode = 0
    
    try:
        while not stop_event.is_set():
            try:
                while True:
                    episode_data = experience_queue.get_nowait()
                    replay_buffer.extend(episode_data)
                    episode += 1
                    
                    if len(replay_buffer) >= batch_size:
                        batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                        batch_data = [replay_buffer[i] for i in batch_indices]
                        
                        policy_loss, value_loss = optimize_model(model, optimizer, batch_data, XiangqiAgent(model))
                        
                        writer.add_scalar('Loss/Policy_Loss', policy_loss, episode)
                        writer.add_scalar('Loss/Value_Loss', value_loss, episode)
                    
                    if episode % save_interval == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode}.pt")
                        torch.save({
                            'episode': episode,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
            except queue.Empty:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")
        writer.close()

def train(num_episodes=1000, num_processes=4, batch_size=64, visualize=True, pretrained_model=None, max_moves=2000):
    """Main training function using multiple processes"""
    if pretrained_model:
        model = pretrained_model
    else:
        model = XiangqiHybridNet()
    model.share_memory()
    
    experience_queue = Queue()
    stop_event = Event()
    progress_queue = Queue()
    
    # Create a single progress bar for overall training
    pbar = tqdm(total=num_episodes, desc='Training Progress', 
                unit='episodes', ncols=100,
                postfix={'buffer': 0, 'epsilon': 1.0},
                position=0,  # Fix position to 0
                leave=True,  # Keep the progress bar after completion
                dynamic_ncols=True,  # Adapt to terminal width
                mininterval=0.1,  # Minimum progress display update interval
                maxinterval=1.0)  # Maximum progress display update interval
    
    # Start training process
    trainer = mp.Process(target=train_model, 
                        args=(model, experience_queue, stop_event, batch_size))
    trainer.start()
    
    # Start game playing processes
    players = []
    episodes_per_process = num_episodes // num_processes
    for rank in range(num_processes):
        process_visualize = visualize and rank == 0
        p = mp.Process(target=play_games,
                      args=(rank, model, experience_queue, stop_event, 
                            episodes_per_process, max_moves, process_visualize,
                            progress_queue))
        p.start()
        players.append(p)
    
    try:
        # Monitor progress
        completed_episodes = 0
        while completed_episodes < num_episodes:
            try:
                # Update progress bar for each completed episode
                progress = progress_queue.get(timeout=1.0)
                completed_episodes += progress
                
                # Update postfix info periodically
                if completed_episodes % 10 == 0:
                    epsilon = max(0.1, 1.0 * (1 - completed_episodes/num_episodes))
                    pbar.set_postfix_str(
                        f"completed={completed_episodes}/{num_episodes}, epsilon={epsilon:.2f}",
                        refresh=False  # Don't refresh immediately
                    )
                
                pbar.update(progress)  # This will refresh the display
                
            except queue.Empty:
                if all(not p.is_alive() for p in players):
                    break
                continue
        
        for p in players:
            p.join()
        
        stop_event.set()
        trainer.join()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        stop_event.set()
        for p in players:
            p.terminate()
        trainer.terminate()
    finally:
        pbar.close()
    
    return model

def optimize_model(model, optimizer, batch_data, agent):
    states, actions, rewards = zip(*batch_data)
    
    # Get the device from the model
    device = next(model.parameters()).device
    
    # Move tensors to the correct device
    states_array = np.array(states)
    state_tensor = torch.FloatTensor(states_array).to(device)
    action_tensor = torch.LongTensor([agent._move_to_index(a) for a in actions]).to(device)
    reward_tensor = torch.FloatTensor(rewards).to(device)
    
    # Calculate loss
    policy, value = model(state_tensor)
    policy_loss = -torch.mean(policy.gather(1, action_tensor.unsqueeze(1)))
    value_loss = torch.mean((value - reward_tensor) ** 2)
    
    loss = policy_loss + value_loss
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return policy_loss.item(), value_loss.item()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='Whether to pretrain on database')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize training')
    parser.add_argument('--episodes', type=int, default=64000, help='Number of episodes to train')
    parser.add_argument('--max-moves', type=int, default=1000, help='Maximum moves per game')
    parser.add_argument('--processes', type=int, 
                       default=max(1, multiprocessing.cpu_count() // 2),
                       help='Number of game playing processes (default: half of CPU cores)')
    args = parser.parse_args()

    # Enable multiprocessing support for CUDA
    mp.set_start_method('spawn')

    logger.info(f"Using {args.processes} processes for training")

    model = XiangqiHybridNet()
    
    if args.pretrain:
        # You would need to provide a path to your games database
        database_path = "xiangqi_games.csv"
        try:
            pretrained_model = pretrain_on_database(model, database_path)
            logger.info("Pretraining completed. Starting reinforcement learning...")
        except FileNotFoundError:
            logger.warning("No game database found. Starting with untrained model...")
            pretrained_model = model
    else:
        logger.info("Skipping pretraining. Starting with untrained model...")
        pretrained_model = model
    
    # Then fine-tune with reinforcement learning using multiple processes
    train(num_episodes=args.episodes, 
          num_processes=args.processes,
          visualize=args.visualize, 
          pretrained_model=pretrained_model if args.pretrain else model,
          max_moves=args.max_moves)