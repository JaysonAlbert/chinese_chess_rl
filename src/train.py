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
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    # 创建回放缓冲区
    replay_buffer = []
    max_buffer_size = 100000  # 设置缓冲区最大容量
    
    # 添加epsilon-greedy探索
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    if visualize:
        vis = XiangqiVisualizer(env)
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track best model
    best_reward = float('-inf')
    best_win_rate = 0.0
    
    try:
        episode_rewards = []
        episode_lengths = []
        running_reward = 0
        
        # 添加统计计数器
        stats = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_captures': 0,
            'pieces_captured': {piece: 0 for piece in ['p', 'c', 'h', 'r', 'a', 'e', 'k']},
            'moves_to_win': [],
            'checks_made': 0,
            'river_crosses': 0
        }
        
        # 添加损失统计
        running_policy_loss = 0
        running_value_loss = 0
        episode_policy_losses = []
        episode_value_losses = []
        
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            done = False
            episode_data = []
            move_count = 0
            episode_reward = 0
            episode_captures = 0
            episode_checks = 0
            episode_crosses = 0
            policy_losses = []
            value_losses = []
            
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
                
                # 使用epsilon-greedy策略选择动作
                if random.random() < epsilon:
                    action = random.choice(valid_moves)
                else:
                    action = agent.select_action(state_array, valid_moves, temperature=1.0)
                
                next_state, reward, done = env.step(action)
                
                # 保存经验到episode_data
                episode_data.append((state_array, action, reward))
                episode_reward += reward
                
                # 统计当前回合的数据
                if env.last_move:
                    # 记录吃子
                    if 'captured' in env.history[-1] and env.history[-1]['captured']:
                        episode_captures += 1
                        captured_piece = env.history[-1]['captured']
                        stats['pieces_captured'][captured_piece] += 1
                        stats['total_captures'] += 1
                    
                    # 记录过河
                    to_row = env.last_move[1][0]
                    if (to_row < 5 and env.current_player) or (to_row > 4 and not env.current_player):
                        episode_crosses += 1
                
                # 记录将军
                if env._is_in_check():
                    episode_checks += 1
                
                state = next_state
                move_count += 1
            
            # 更新游戏结果统计
            if env.is_game_over:
                if env.winner is True:  # 红方胜
                    stats['wins'] += 1
                    stats['moves_to_win'].append(move_count)
                elif env.winner is False:  # 黑方胜
                    stats['losses'] += 1
                else:  # 和棋
                    stats['draws'] += 1
            
            # 更新统计
            stats['checks_made'] += episode_checks
            stats['river_crosses'] += episode_crosses
            
            # Log detailed metrics
            writer.add_scalar('Game/Wins', stats['wins'], episode)
            writer.add_scalar('Game/Losses', stats['losses'], episode)
            writer.add_scalar('Game/Draws', stats['draws'], episode)
            writer.add_scalar('Game/Win_Rate', stats['wins']/(episode+1), episode)
            
            writer.add_scalar('Actions/Captures_Per_Episode', episode_captures, episode)
            writer.add_scalar('Actions/Checks_Per_Episode', episode_checks, episode)
            writer.add_scalar('Actions/River_Crosses_Per_Episode', episode_crosses, episode)
            
            # Log piece-specific capture statistics
            for piece, count in stats['pieces_captured'].items():
                writer.add_scalar(f'Captures/{piece}_captured', count, episode)
            
            # Log average moves to win
            if stats['moves_to_win']:
                avg_moves_to_win = sum(stats['moves_to_win']) / len(stats['moves_to_win'])
                writer.add_scalar('Game/Avg_Moves_To_Win', avg_moves_to_win, episode)
            
            # 衰减epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # 更新经验回放 - 现在episode_data不再为空
            replay_buffer.extend(episode_data)
            if len(replay_buffer) > max_buffer_size:
                replay_buffer = replay_buffer[-max_buffer_size:]
            
            # 增加训练频率
            if len(replay_buffer) >= batch_size:
                num_training_iterations = 8
                for _ in range(num_training_iterations):
                    batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                    batch_data = [replay_buffer[i] for i in batch_indices]
                    p_loss, v_loss = optimize_model(model, optimizer, batch_data, agent)
                    policy_losses.append(p_loss)
                    value_losses.append(v_loss)
            
            # 计算本回合的平均损失
            if policy_losses:  # 如果这回合有训练
                avg_policy_loss = np.mean(policy_losses)
                avg_value_loss = np.mean(value_losses)
                episode_policy_losses.append(avg_policy_loss)
                episode_value_losses.append(avg_value_loss)
                
                # 更新运行平均损失
                running_policy_loss = 0.95 * running_policy_loss + 0.05 * avg_policy_loss if episode > 0 else avg_policy_loss
                running_value_loss = 0.95 * running_value_loss + 0.05 * avg_value_loss if episode > 0 else avg_value_loss
                
                # Log loss metrics
                writer.add_scalar('Loss/Policy_Loss', avg_policy_loss, episode)
                writer.add_scalar('Loss/Value_Loss', avg_value_loss, episode)
                writer.add_scalar('Loss/Running_Policy_Loss', running_policy_loss, episode)
                writer.add_scalar('Loss/Running_Value_Loss', running_value_loss, episode)
                
                # 计算并记录最近100回合的平均损失
                if len(episode_policy_losses) >= 100:
                    avg_policy_loss_100 = np.mean(episode_policy_losses[-100:])
                    avg_value_loss_100 = np.mean(episode_value_losses[-100:])
                    writer.add_scalar('Loss/Average_Policy_Loss_100', avg_policy_loss_100, episode)
                    writer.add_scalar('Loss/Average_Value_Loss_100', avg_value_loss_100, episode)
            
            # Update statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(move_count)
            running_reward = 0.95 * running_reward + 0.05 * episode_reward if episode > 0 else episode_reward
            
            # Log exploration and training metrics
            writer.add_scalar('Training/Epsilon', epsilon, episode)
            writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
            writer.add_scalar('Training/Running_Reward', running_reward, episode)
            writer.add_scalar('Training/Episode_Length', move_count, episode)
            writer.add_scalar('Training/Buffer_Size', len(replay_buffer), episode)
            
            # Log periodic statistics
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                
                writer.add_scalar('Training/Average_Reward_100', avg_reward, episode)
                writer.add_scalar('Training/Average_Length_100', avg_length, episode)
                
                # 添加详细日志
                logger.info(f"Episode {episode}")
                logger.info(f"Running reward: {running_reward:.2f}")
                logger.info(f"Win rate: {stats['wins']/(episode+1):.2%}")
                logger.info(f"Total captures: {stats['total_captures']}")
                logger.info(f"Average moves to win: {avg_moves_to_win:.1f}" if stats['moves_to_win'] else "No wins yet")
                logger.info(f"Buffer size: {len(replay_buffer)}")
                logger.info(f"Epsilon: {epsilon:.3f}")
                
                # 添加损失信息到日志
                if policy_losses:
                    logger.info(f"Policy Loss: {running_policy_loss:.4f}")
                    logger.info(f"Value Loss: {running_value_loss:.4f}")
                    if len(episode_policy_losses) >= 100:
                        logger.info(f"Avg Policy Loss (100 ep): {avg_policy_loss_100:.4f}")
                        logger.info(f"Avg Value Loss (100 ep): {avg_value_loss_100:.4f}")
            
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
            
            # Save periodic checkpoints (every 1000 episodes)
            if episode > 0 and episode % 1000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode}.pt")
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'running_reward': running_reward,
                    'epsilon': epsilon,
                    'stats': stats,
                    'replay_buffer': replay_buffer,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model based on running reward
            if running_reward > best_reward:
                best_reward = running_reward
                best_model_path = os.path.join(checkpoint_dir, "best_reward_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best reward model saved: {running_reward:.2f}")
            
            # Save best model based on win rate
            current_win_rate = stats['wins']/(episode+1)
            if current_win_rate > best_win_rate and episode > 100:
                best_win_rate = current_win_rate
                best_model_path = os.path.join(checkpoint_dir, "best_winrate_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best win-rate model saved: {current_win_rate:.2%}")
            
    except KeyboardInterrupt:
        # Save final model on interrupt
        logger.info("\nTraining interrupted, saving checkpoint...")
        interrupt_path = os.path.join(checkpoint_dir, "interrupted_model.pt")
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'running_reward': running_reward,
            'epsilon': epsilon,
            'stats': stats,
            'replay_buffer': replay_buffer,
        }, interrupt_path)
        logger.info(f"Saved interrupt checkpoint to {interrupt_path}")
    finally:
        # Save final model
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")
        
        if visualize:
            vis.close()
        writer.close()
    
    return model

def optimize_model(model, optimizer, batch_data, agent):
    states, actions, rewards = zip(*batch_data)
    
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
    
    return policy_loss.item(), value_loss.item()

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