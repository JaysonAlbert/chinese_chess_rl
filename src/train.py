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

def train(num_episodes=1000, batch_size=64, visualize=True, pretrained_model=None, max_moves=2000):
    env = XiangqiEnv()
    
    # Set up TensorBoard writer
    log_dir = "logs/tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Load pretrained model or create new one
    if pretrained_model:
        model = pretrained_model
    else:
        model = XiangqiHybridNet()
        
    agent = XiangqiAgent(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 创建回放缓冲区
    replay_buffer = []
    max_buffer_size = 100000  # 设置缓冲区最大容量
    
    if visualize:
        vis = XiangqiVisualizer(env)
    
    try:
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            done = False
            episode_data = []
            move_count = 0
            
            while not done and move_count < max_moves:
                if visualize:
                    vis.draw_board()
                    pygame.time.wait(100)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                    
                state_array = env._get_state()
                action = agent.select_action(state_array, valid_moves, temperature=1.0)
                next_state, reward, done = env.step(action)
                episode_data.append((state_array, action, reward))
                
                state = next_state
                move_count += 1
            
            # 将本局的数据添加到回放缓冲区
            replay_buffer.extend(episode_data)
            
            # 如果超出最大容量，移除最早的数据
            if len(replay_buffer) > max_buffer_size:
                replay_buffer = replay_buffer[-max_buffer_size:]
            
            # 只要缓冲区中的数据量足够一个batch，就进行多次训练
            if len(replay_buffer) >= batch_size:
                num_training_iterations = 4  # 每局结束后的训练次数
                for _ in range(num_training_iterations):
                    # 随机采样一个batch的数据
                    batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                    batch_data = [replay_buffer[i] for i in batch_indices]
                    optimize_model(model, optimizer, batch_data, agent)
            
            # Log metrics
            writer.add_scalar('Game/Moves_Per_Episode', move_count, episode)
            writer.add_scalar('Training/Buffer_Size', len(replay_buffer), episode)
            
            if move_count >= max_moves:
                logger.info(f"Episode {episode} reached move limit of {max_moves}")
                # Create a temporary visualizer just for saving the board state
                temp_vis = XiangqiVisualizer(env)
                temp_vis.draw_board()
                
                # Save the board image to dir logs/images
                save_dir = "logs/images"
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"long_game_episode_{episode}.png")
                pygame.image.save(temp_vis.screen, save_path)
                logger.info(f"Saved board state to {save_path}")
                
                # Clean up the temporary visualizer
                temp_vis.close()
                if visualize:
                    vis = XiangqiVisualizer(env)  # Recreate the main visualizer
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted")
    finally:
        if visualize:
            vis.close()
        writer.close()
    
    return model

def optimize_model(model, optimizer, batch_data, agent):
    states, actions, rewards = zip(*batch_data)
    
    # Convert to numpy array first
    states_array = np.array(states)
    state_tensor = torch.FloatTensor(states_array)
    action_tensor = torch.LongTensor([agent._move_to_index(a) for a in actions])
    reward_tensor = torch.FloatTensor(rewards)
    
    # Calculate loss
    policy, value = model(state_tensor)
    policy_loss = -torch.mean(policy.gather(1, action_tensor.unsqueeze(1)))
    value_loss = torch.mean((value - reward_tensor) ** 2)
    
    loss = policy_loss + value_loss
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='Whether to pretrain on database')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize training')
    parser.add_argument('--episodes', type=int, default=64000, help='Number of episodes to train')
    parser.add_argument('--max-moves', type=int, default=1000, help='Maximum moves per game')
    args = parser.parse_args()

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
    
    # Then fine-tune with reinforcement learning
    train(num_episodes=args.episodes, visualize=args.visualize, 
          pretrained_model=pretrained_model, max_moves=args.max_moves)