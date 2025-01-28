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
import threading
from queue import Empty

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
    def __init__(self, num_episodes=1000, batch_size=256, max_moves=2000):
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.max_moves = max_moves
        self.training_iterations_per_episode = 4
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.max_buffer_size = 500000
        self.checkpoint_interval = 1000
        self.log_interval = 10
        self.gamma = 0.99

class TrainingStats:
    """Track training statistics"""
    def __init__(self):
        self.red_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_captures = 0
        self.checks_made = 0
        self.river_crosses = 0
        self.moves_to_win = []
        self.pieces_captured = {piece: 0 for piece in ['p', 'c', 'h', 'r', 'a', 'e', 'k']}
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.running_reward = 0
        self.best_reward = float('-inf')
        self.best_win_rate = 0.0
        self.total_games = 0
        self.running_win_rate = 0.0

    def update_win_rate(self):
        """Calculate current win rate"""
        if self.total_games > 0:
            return self.red_wins / self.total_games
        return 0.0

class TrainingManager:
    """Manages the training process"""
    def __init__(self, config, model, visualize=False):
        self.config = config
        self.model = model.to(get_device())
        self.agent = XiangqiAgent(model)
        self.optimizer = optim.Adam(model.parameters(), lr=0.0003)
        self.stats = TrainingStats()
        self.replay_buffer = []
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.writer = self._setup_tensorboard()
        self.visualize = visualize
        self.vis = XiangqiVisualizer(XiangqiEnv()) if visualize else None
        
    def _setup_tensorboard(self):
        log_dir = "logs/tensorboard"
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)

    def train(self, use_multiprocess=False):
        """Main training loop"""
        try:
            self._train_singleprocess()
        except KeyboardInterrupt:
            self._handle_interrupt()
        finally:
            self._cleanup()

    def _train_singleprocess(self):
        env = XiangqiEnv()
        pbar = tqdm(range(self.config.num_episodes), desc='Training (SP)')
        
        for episode in pbar:
            result = self._run_episode(env)
            self._process_episode_result(result, episode)
            self._periodic_logging(episode)
            self._save_checkpoints(episode)

    def _process_episode_result(self, result, episode):
        """Process results from a single episode"""
        try:
            episode_data, move_count, is_game_over, final_reward, info = result
            
            # Update replay buffer with thread safety
            with threading.Lock():
                self.replay_buffer.extend(episode_data)
                if len(self.replay_buffer) > self.config.max_buffer_size:
                    self.replay_buffer = self.replay_buffer[-self.config.max_buffer_size:]
            
            # Train on batches
            if len(self.replay_buffer) >= self.config.batch_size:
                self._train_on_batch()
            
            # Update statistics
            self._update_stats(final_reward, move_count, is_game_over, info)
            
            # Log metrics
            self._log_metrics(episode, final_reward, move_count)
        except Exception as e:
            logger.error(f"Error processing episode result: {e}")

    def _train_on_batch(self):
        """Perform one training iteration on a batch of data"""
        indices = np.random.choice(len(self.replay_buffer), 
                                 self.config.batch_size, replace=False)
        batch_data = [self.replay_buffer[i] for i in indices]
        
        p_loss, v_loss = optimize_model_mixed_precision(
            self.model, self.optimizer, batch_data, 
            self.agent, self.scaler
        )
        
        self.stats.policy_losses.append(p_loss)
        self.stats.value_losses.append(v_loss)

    def _update_stats(self, final_reward, move_count, is_game_over, info):
        """Update training statistics"""
        self.stats.running_reward = (0.95 * self.stats.running_reward + 
                                   0.05 * final_reward)
        
        # Increment total games when a game ends
        self.stats.total_games += 1
        
        if info['winner'] == True:
            self.stats.red_wins += 1
        elif info['winner'] == False:
            self.stats.black_wins += 1
        else:
            self.stats.draws += 1
            
            # Update running win rate
        self.stats.running_win_rate = self.stats.update_win_rate()

    def _cleanup(self):
        """Cleanup resources"""
        if self.visualize and self.vis:
            self.vis.close()
        self.writer.close()

    def _run_episode(self, env, agent=None):
        """Run a single episode"""
        state = env.reset()
        episode_data = []
        move_count = 0
        episode_reward = 0
        
        for move_count in range(self.config.max_moves):
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                # Game over - current player has no valid moves (checkmate or stalemate)
                if env.is_in_check():
                    episode_reward = -10.0  # Lost by checkmate
                else:
                    episode_reward = 0.0  # Draw by stalemate
                break
                
            if env.is_game_over:
                if env.is_in_check():
                    episode_reward = 10.0  # Won by checkmate
                break
                
            if agent is None:
                action = self.agent.select_action(
                    torch.FloatTensor(env._get_state()).to(self.model.device), 
                    valid_moves, 
                    temperature=1.0
                )
            else:
                action = agent.select_action(
                    torch.FloatTensor(env._get_state()), 
                    valid_moves, 
                    temperature=1.0
                )
            next_state, reward, done, info = env.step(action)
            
            if reward != 0:  # If a capture or check occurred
                episode_reward += reward
            
            episode_data.append((env._get_state(), action, reward, done))
            state = next_state
            
            if done:
                logger.info(f"Game over. Episode reward: {episode_reward}, Winner: {info['winner']}")
                break
        
        return episode_data, move_count, env.is_game_over, episode_reward, info

    def _periodic_logging(self, episode):
        """Log periodic training statistics"""
        if episode % self.config.log_interval == 0:
            avg_reward = np.mean(self.stats.episode_rewards[-self.config.log_interval:]) if self.stats.episode_rewards else 0
            avg_length = np.mean(self.stats.episode_lengths[-self.config.log_interval:]) if self.stats.episode_lengths else 0
            
            self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
            self.writer.add_scalar('Training/Average_Length', avg_length, episode)
            self.writer.add_scalar('Training/Win_Rate', self.stats.running_win_rate, episode)
            
            logger.info(f"Episode {episode}")
            logger.info(f"Running reward: {self.stats.running_reward:.2f}")
            logger.info(f"Win rate: {self.stats.running_win_rate:.2%}")
            logger.info(f"Total games: {self.stats.total_games}")
            logger.info(f"Wins/Losses/Draws: {self.stats.red_wins}/{self.stats.black_wins}/{self.stats.draws}")
            logger.info(f"Buffer size: {len(self.replay_buffer)}")
            
            if self.stats.policy_losses:
                avg_policy_loss = np.mean(self.stats.policy_losses[-self.config.log_interval:])
                avg_value_loss = np.mean(self.stats.value_losses[-self.config.log_interval:])
                self.writer.add_scalar('Loss/Policy_Loss', avg_policy_loss, episode)
                self.writer.add_scalar('Loss/Value_Loss', avg_value_loss, episode)

    def _save_checkpoints(self, episode):
        """Save periodic checkpoints"""
        if episode > 0 and episode % self.config.checkpoint_interval == 0:
            checkpoint_dir = "logs/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode}.pt")
            
            torch.save({
                'episode': episode,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'running_reward': self.stats.running_reward,
                'stats': self.stats.__dict__,
                'replay_buffer': self.replay_buffer,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _handle_interrupt(self):
        """Handle training interruption"""
        logger.info("\nTraining interrupted, saving checkpoint...")
        checkpoint_dir = "logs/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        interrupt_path = os.path.join(checkpoint_dir, "interrupted_model.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_reward': self.stats.running_reward,
            'stats': self.stats.__dict__,
            'replay_buffer': self.replay_buffer,
        }, interrupt_path)
        logger.info(f"Saved interrupt checkpoint to {interrupt_path}")

    def _log_metrics(self, episode, final_reward, move_count):
        """Log training metrics"""
        self.stats.episode_rewards.append(final_reward)
        self.stats.episode_lengths.append(move_count)
        
        self.writer.add_scalar('Training/Episode_Reward', final_reward, episode)
        self.writer.add_scalar('Training/Running_Reward', self.stats.running_reward, episode)
        self.writer.add_scalar('Training/Episode_Length', move_count, episode)
        self.writer.add_scalar('Training/Buffer_Size', len(self.replay_buffer), episode)

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
        next_return[t] = returns[t]
    
    return returns.unsqueeze(1)  # Add back the dimension for consistency

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
    config = TrainingConfig(
        num_episodes=args.episodes,
        max_moves=args.max_moves
    )
    trainer = TrainingManager(config, pretrained_model, visualize=args.visualize)
    trainer.train(use_multiprocess=args.multiprocess)