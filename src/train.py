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
from xiangqi import Move

def pretrain_on_database(model, database_path, num_epochs=10, batch_size=64):
    """Pretrain the model on human games database"""
    print("Loading game database...")
    games_df = pd.read_csv(database_path)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Can use higher learning rate with hybrid
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for _, game in tqdm(games_df.iterrows(), total=len(games_df)):
            env = XiangqiEnv()
            moves = game['moves'].split()  # Assuming moves are stored in algebraic notation
            
            for move_str in moves:
                # Convert move string to our format
                from_pos = (int(move_str[1]), int(move_str[0]))
                to_pos = (int(move_str[3]), int(move_str[2]))
                
                state = env._get_state()
                
                # Get model's prediction
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy, value = model(state_tensor)
                
                # Calculate target
                target_move = env._move_to_index((from_pos, to_pos))
                target_move_tensor = torch.LongTensor([target_move])
                
                # Calculate loss
                policy_loss = F.cross_entropy(policy, target_move_tensor)
                
                # Update model
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
                
                total_loss += policy_loss.item()
                num_batches += 1
                
                # Make the move
                env.step((from_pos, to_pos))
        
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
                    
                action = agent.select_action(state, valid_moves, temperature=1.0)
                next_state, reward, done = env.step(action)
                episode_data.append((state, action, reward))
                
                state = next_state
            
            if len(episode_data) >= batch_size:
                optimize_model(model, optimizer, episode_data[-batch_size:])
                
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        if visualize:
            pygame.quit()
    
    return model

def optimize_model(model, optimizer, batch_data):
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
    # First pretrain on database
    model = XiangqiHybridNet()
    
    # You would need to provide a path to your games database
    database_path = "xiangqi_games.csv"
    try:
        pretrained_model = pretrain_on_database(model, database_path)
        print("Pretraining completed. Starting reinforcement learning...")
    except FileNotFoundError:
        print("No game database found. Starting with untrained model...")
        pretrained_model = model
    
    # Then fine-tune with reinforcement learning
    train(num_episodes=100, visualize=True, pretrained_model=pretrained_model) 