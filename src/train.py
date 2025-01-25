import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pygame
from environment import XiangqiEnv
from agent import XiangqiAgent
from model import XiangqiHybridNet
from visualize import XiangqiVisualizer
import pandas as pd

def pretrain_on_database(model, database_path, num_epochs=10, batch_size=64):
    """Pretrain the model on human games database"""
    print("Loading game database...")
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
                        state_tensor = torch.FloatTensor(states)
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
                    print(f"跳过无效的移动: {move_str}, 错误: {e}")
                    continue
            
            # 处理剩余的数据
            if states:
                state_tensor = torch.FloatTensor(states)
                target_moves_tensor = torch.LongTensor(target_moves)
                
                policy, value = model(state_tensor)
                policy_loss = F.cross_entropy(policy, target_moves_tensor)
                
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
                
                total_loss += policy_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def train(num_episodes=1000, batch_size=64, visualize=True, pretrained_model=None):
    env = XiangqiEnv()
    
    # Load pretrained model or create new one
    if pretrained_model:
        model = pretrained_model
    else:
        model = XiangqiHybridNet()
        
    agent = XiangqiAgent(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    
    if visualize:
        vis = XiangqiVisualizer(env)
    
    try:
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            done = False
            episode_data = []
            
            while not done:
                if visualize:
                    vis.draw_board()
                    pygame.time.wait(100)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                    
                state_array = env._get_state()  # Get numerical state representation
                action = agent.select_action(state_array, valid_moves, temperature=1.0)
                next_state, reward, done = env.step(action)
                episode_data.append((state_array, action, reward))
                
                state = next_state
            
            if len(episode_data) >= batch_size:
                optimize_model(model, optimizer, episode_data[-batch_size:], agent)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        if visualize:
            pygame.quit()
    
    return model

def optimize_model(model, optimizer, batch_data, agent):
    states, actions, rewards = zip(*batch_data)
    
    state_tensor = torch.FloatTensor(states)
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
    args = parser.parse_args()

    model = XiangqiHybridNet()
    
    if args.pretrain:
        # You would need to provide a path to your games database
        database_path = "xiangqi_games.csv"
        try:
            pretrained_model = pretrain_on_database(model, database_path)
            print("Pretraining completed. Starting reinforcement learning...")
        except FileNotFoundError:
            print("No game database found. Starting with untrained model...")
            pretrained_model = model
    else:
        print("Skipping pretraining. Starting with untrained model...")
        pretrained_model = model
    
    # Then fine-tune with reinforcement learning
    train(num_episodes=100, visualize=True, pretrained_model=pretrained_model)