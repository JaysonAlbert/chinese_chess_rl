import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pygame
import logging
import numpy as np
from xiangqi_rl.environment import XiangqiEnv
from xiangqi_rl.agent import XiangqiAgent
from xiangqi_rl.model import XiangqiHybridNet, get_device
from xiangqi_rl.visualize import XiangqiVisualizer
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import random
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
import torch.distributed as dist
from torch.amp import GradScaler
from contextlib import nullcontext

# Set start method to spawn
if __name__ == '__main__':
    mp.set_start_method('spawn')

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

def run_episode(args):
    """Run one episode in parallel"""
    env_idx, env, agent, max_moves = args
    state = env.reset()
    episode_data = []
    move_count = 0
    episode_checks = 0
    episode_crosses = 0
    episode_captures = 0
    episode_reward = 0
    
    # Pre-allocate state tensor to avoid repeated creation
    device = next(agent.model.parameters()).device
    state_tensor = torch.zeros((1, 14, 10, 9), device=device)
    
    for move_count in range(max_moves):
        valid_moves = env.get_valid_moves()
        if not valid_moves or env.is_game_over:
            break
        
        # Reuse tensor instead of creating new one
        state_tensor[0] = torch.FloatTensor(env._get_state())
        action = agent.select_action(state_tensor[0], valid_moves, temperature=1.0)
        step_result = env.step(action)
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
        else:
            next_state, reward, done = step_result
        
        episode_data.append((env._get_state(), action, reward, done))
        state = next_state
        episode_reward += reward
        
        if done:
            break
        
        # Track stats
        if env._is_in_check():
            episode_checks += 1
        if env.last_move:
            to_row = env.last_move[1][0]
            if (to_row < 5 and env.current_player) or (to_row > 4 and not env.current_player):
                episode_crosses += 1
            
            # Check if a capture occurred by looking at the history
            if len(env.history) > 0 and 'captured' in env.history[-1]:
                if env.history[-1]['captured']:
                    episode_captures += 1
    
    # Ensure we return a valid reward even if game isn't finished
    final_reward = episode_reward
    if env.is_game_over:
        if env.winner is True:  # Red wins
            final_reward += 1.0
        elif env.winner is False:  # Black wins
            final_reward -= 1.0
    
    return episode_data, move_count, env.is_game_over, final_reward, episode_checks, episode_crosses, episode_captures, episode_reward

def train(num_episodes=1000, batch_size=256, visualize=True, pretrained_model=None, max_moves=2000, use_multiprocess=False):
    # Increase batch size for better GPU utilization
    env = XiangqiEnv()
    
    # Create checkpoint directory
    checkpoint_dir = "logs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training parameters
    training_iterations_per_episode = 4  # Reduced from 16 for better performance
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    # Set up TensorBoard writer
    log_dir = "logs/tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize scaler only if CUDA is available
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Increase number of worker threads for data loading
    num_workers = mp.cpu_count()
    
    # Set up device and model
    device = get_device()
    logger.info(f"Using device: {device}")
    if pretrained_model:
        model = pretrained_model.to(device)
    else:
        model = XiangqiHybridNet().to(device)
    
    # Enable model parallelism if model is large enough
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Make sure model is on correct device
    model = model.to(device)
    agent = XiangqiAgent(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    # Increase replay buffer size for better training
    replay_buffer = []
    max_buffer_size = 500000  # Increased from 100000
    
    # Initialize tracking variables
    episode_rewards = []
    episode_lengths = []
    running_reward = 0
    policy_losses = []
    value_losses = []
    running_policy_loss = 0
    running_value_loss = 0
    episode_policy_losses = []
    episode_value_losses = []
    
    # Initialize stats
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
    
    # Track best model performance
    best_reward = float('-inf')
    best_win_rate = 0.0
    
    # Initialize visualizer if needed
    vis = XiangqiVisualizer(env) if visualize else None
    
    try:
        if use_multiprocess:
            # Multiprocess version
            num_envs = min(mp.cpu_count() // 2, 8)
            envs = [XiangqiEnv() for _ in range(num_envs)]
            manager = mp.Manager()
            replay_buffer = manager.list()
            
            with mp.Pool(num_envs) as pool:
                pbar = tqdm(total=num_episodes, desc='Training (MP)')
                for episode in range(0, num_episodes, num_envs):
                    # Run episodes in parallel
                    args_list = [(i, envs[i], agent, max_moves) for i in range(num_envs)]
                    episode_results = pool.map(run_episode, args_list)
                    
                    # Batch process results
                    all_episode_data = []
                    for result in episode_results:
                        all_episode_data.extend(result[0])  # Collect all episode data
                    
                    # Train on larger batches less frequently
                    if len(replay_buffer) >= batch_size * 2:  # Ensure enough data
                        for _ in range(training_iterations_per_episode):
                            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                            batch_data = [replay_buffer[i] for i in indices]
                            p_loss, v_loss = optimize_model_mixed_precision(model, optimizer, batch_data, agent, scaler)
                            policy_losses.append(p_loss)
                            value_losses.append(v_loss)
                    
                    # Update buffer after training
                    replay_buffer.extend(all_episode_data)
                    if len(replay_buffer) > max_buffer_size:
                        del replay_buffer[:len(replay_buffer)-max_buffer_size]
                    
                    # Update statistics
                    episode_lengths.append(result[1])
                    episode_reward = result[3] if result[3] is not None else 0
                    running_reward = 0.95 * running_reward + 0.05 * episode_reward if episode > 0 else episode_reward
                    
                    # Log exploration and training metrics
                    writer.add_scalar('Training/Epsilon', epsilon, episode)
                    writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
                    writer.add_scalar('Training/Running_Reward', running_reward, episode)
                    writer.add_scalar('Training/Episode_Length', result[1], episode)
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
                                logger.info(f"Avg Policy Loss (100 ep): {np.mean(episode_policy_losses[-100:]):.4f}")
                                logger.info(f"Avg Value Loss (100 ep): {np.mean(episode_value_losses[-100:]):.4f}")
                    
                    if result[1] >= max_moves:
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
                    
                    # Update statistics
                    stats['checks_made'] += result[4]
                    stats['river_crosses'] += result[5]
                    stats['total_captures'] += result[6]
                    if result[2]:
                        if result[3] is True:  # 红方胜
                            stats['wins'] += 1
                            stats['moves_to_win'].append(result[1])
                        elif result[3] is False:  # 黑方胜
                            stats['losses'] += 1
                        else:  # 和棋
                            stats['draws'] += 1
                    
                    # Log detailed metrics
                    writer.add_scalar('Game/Wins', stats['wins'], episode)
                    writer.add_scalar('Game/Losses', stats['losses'], episode)
                    writer.add_scalar('Game/Draws', stats['draws'], episode)
                    writer.add_scalar('Game/Win_Rate', stats['wins']/(episode+1), episode)
                    
                    writer.add_scalar('Actions/Captures_Per_Episode', result[6], episode)
                    writer.add_scalar('Actions/Checks_Per_Episode', result[4], episode)
                    writer.add_scalar('Actions/River_Crosses_Per_Episode', result[5], episode)
                    
                    # Log piece-specific capture statistics
                    for piece, count in stats['pieces_captured'].items():
                        writer.add_scalar(f'Captures/{piece}_captured', count, episode)
                    
                    # Log average moves to win
                    if stats['moves_to_win']:
                        avg_moves_to_win = sum(stats['moves_to_win']) / len(stats['moves_to_win'])
                        writer.add_scalar('Game/Avg_Moves_To_Win', avg_moves_to_win, episode)
                    
                    # 衰减epsilon
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)
                    
                    # 更新运行平均损失
                    if policy_losses:
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
                    
                    pbar.update(num_envs)
        else:
            # Single process version
            pbar = tqdm(total=num_episodes, desc='Training (SP)')
            for episode in range(num_episodes):
                state = env.reset()
                episode_data = []
                move_count = 0
                episode_checks = 0
                episode_crosses = 0
                episode_captures = 0
                episode_reward = 0

                for move_count in range(max_moves):
                    if visualize and vis is not None:
                        vis.draw_board()
                        if env.last_move:  # Only animate if there was a previous move
                            from_pos, to_pos = env.last_move
                            vis.animate_move(from_pos, to_pos)
                        pygame.time.wait(100)

                    valid_moves = env.get_valid_moves()
                    if not valid_moves or env.is_game_over:
                        break

                    action = agent.select_action(env._get_state(), valid_moves, temperature=1.0)
                    step_result = env.step(action)
                    if len(step_result) == 4:
                        next_state, reward, done, info = step_result
                    else:
                        next_state, reward, done = step_result
                    
                    episode_data.append((env._get_state(), action, reward, done))
                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                    # Track stats
                    if env._is_in_check():
                        episode_checks += 1
                    if env.last_move:
                        to_row = env.last_move[1][0]
                        if (to_row < 5 and env.current_player) or (to_row > 4 and not env.current_player):
                            episode_crosses += 1
                    
                    # Train on episode data
                    if len(replay_buffer) >= batch_size:
                        for _ in range(training_iterations_per_episode):
                            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                            batch_data = [replay_buffer[i] for i in indices]
                            p_loss, v_loss = optimize_model_mixed_precision(model, optimizer, batch_data, agent, scaler)
                            policy_losses.append(p_loss)
                            value_losses.append(v_loss)

                    # Update replay buffer
                    replay_buffer.extend(episode_data)
                    if len(replay_buffer) > max_buffer_size:
                        del replay_buffer[:len(replay_buffer)-max_buffer_size]

                    # Rest of the existing statistics and logging code...
                    pbar.update(1)
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
        
        if visualize and vis is not None:
            vis.close()
        writer.close()
    
    return model

def optimize_model_mixed_precision(model, optimizer, batch_data, agent, scaler):
    device = next(model.parameters()).device
    states, actions, rewards, dones = zip(*batch_data)  # Add dones to batch data
    
    # Convert to tensors
    state_tensor = torch.FloatTensor(np.array(states)).to(device)  # Convert to numpy array first
    action_tensor = torch.LongTensor([agent._move_to_index(a) for a in actions]).to(device)
    reward_tensor = torch.FloatTensor(rewards).to(device)
    done_tensor = torch.BoolTensor(dones).to(device)
    
    # Only use autocast if CUDA is available
    context = autocast(device_type='cuda') if torch.cuda.is_available() else nullcontext()
    
    with context:
        policy, value = model(state_tensor)
        
        # Compute value targets using n-step returns
        value_targets = compute_value_targets(reward_tensor, value.detach(), done_tensor, gamma=0.99)
        
        # Compute proper value targets using future returns
        value_loss = torch.mean((value - value_targets) ** 2)
        
        # Policy loss using log probabilities
        log_probs = F.log_softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1))
        policy_loss = -(action_log_probs * reward_tensor.unsqueeze(1)).mean()

    # Optimize with gradient scaling if CUDA is available, otherwise normal optimization
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(policy_loss + value_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        (policy_loss + value_loss).backward()
        optimizer.step()
    
    return policy_loss.item(), value_loss.item()

def compute_value_targets(rewards, values, dones, gamma=0.99):
    """
    Compute value targets using n-step returns
    Args:
        rewards: tensor of rewards for the batch
        values: tensor of value predictions
        dones: tensor of done flags
        gamma: discount factor
    """
    batch_size = rewards.size(0)
    returns = torch.zeros_like(rewards)
    
    # Ensure all tensors have the same shape
    values = values.squeeze()  # Remove extra dimensions if any
    
    # Initialize the return with the value of the last state if not done
    next_return = torch.where(dones, torch.zeros_like(values), values)
    
    # Compute returns backwards
    for t in reversed(range(batch_size)):
        returns[t] = rewards[t] + gamma * next_return[t] * (1 - dones[t].float())
    
    return returns.unsqueeze(1)  # Add back the dimension for consistency

def calculate_reward(self, state, action, next_state):
    reward = 0
    # Reward for capturing pieces
    if self.piece_captured:
        reward += 0.1
    # Reward for checking opponent
    if self.is_in_check():
        reward += 0.05
    # Reward for controlling center
    if self.piece_in_center():
        reward += 0.02
    return reward

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        
    def sample(self, batch_size):
        probs = np.array(self.priorities) ** 0.6
        probs = probs / np.sum(probs)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[idx] for idx in indices], indices

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='Whether to pretrain on database')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize training')
    parser.add_argument('--episodes', type=int, default=64000, help='Number of episodes to train')
    parser.add_argument('--max-moves', type=int, default=1000, help='Maximum moves per game')
    parser.add_argument('--multiprocess', action='store_true', help='Whether to use multiprocessing')
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
    train(num_episodes=args.episodes, 
          visualize=args.visualize, 
          pretrained_model=pretrained_model, 
          max_moves=args.max_moves,
          use_multiprocess=args.multiprocess)