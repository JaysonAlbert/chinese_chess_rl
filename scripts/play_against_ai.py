import sys
import os
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from xiangqi_rl.environment import XiangqiEnv
from xiangqi_rl.visualize import XiangqiVisualizer
from xiangqi_rl.model import XiangqiHybridNet
from xiangqi_rl.agent import XiangqiAgent

class AIPlayer:
    def __init__(self, env, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        
        # Initialize the model
        self.model = XiangqiHybridNet().to(device)
        
        if model_path:
            try:
                # Try to load the trained weights with weights_only=True
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model - AI will play randomly")
        else:
            print("No model path provided - Using untrained model")
        
        self.model.eval()
        self.agent = XiangqiAgent(self.model)

    def get_action(self, state):
        """Get the AI's move based on the current state"""
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return None
            
        # Use the agent to select an action
        return self.agent.select_action(state, valid_moves, temperature=1.0)  # Higher temperature for more randomness

def find_model():
    """Find a valid model in the checkpoints directory"""
    checkpoint_dir = os.path.join(project_root, 'logs', 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        return None
        
    # Try different model file patterns
    patterns = [
        'model_episode_*.pt',
        'model_episode_*.pth',
        'best_model.pt',
        'best_model.pth',
        'final_model.pt',
        'final_model.pth'
    ]
    
    for pattern in patterns:
        import glob
        model_files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if model_files:
            return model_files[0]  # Return the first matching model file
            
    return None

def main():
    # Initialize environment and visualizer
    env = XiangqiEnv()
    visualizer = XiangqiVisualizer(env, debug=False)
    
    # Find and load a model
    model_path = find_model()
    if model_path:
        print(f"Found model at: {model_path}")
    else:
        print("No trained model found - AI will play randomly")
    
    # Initialize AI player
    ai_player = AIPlayer(env, model_path)
    
    print("\nGame Controls:")
    print("- Click on a piece to select it")
    print("- Click on a valid destination to move")
    print("- You play as Red (bottom)")
    print("- AI plays as Black (top)")
    print("- Close window to quit\n")
    
    # Game state variables
    selected_pos = None
    valid_moves = []
    game_over = False
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                # Handle human player's move
                if env.current_player:  # Human plays as red (True)
                    mouse_pos = pygame.mouse.get_pos()
                    board_pos = visualizer.get_board_position(mouse_pos)
                    
                    if board_pos is not None:  # Valid board position clicked
                        if selected_pos is None:
                            # Select piece
                            piece = env.board[board_pos[0]][board_pos[1]]
                            if piece and piece.is_red:  # Only allow selecting red pieces
                                selected_pos = board_pos
                                valid_moves = [move for move in env.get_valid_moves() 
                                             if move[0] == selected_pos]
                        else:
                            # Move piece
                            if (selected_pos, board_pos) in valid_moves:
                                # Animate and execute move
                                visualizer.animate_move(selected_pos, board_pos)
                                state, reward, done, info = env.step((selected_pos, board_pos))
                                
                                if done:
                                    game_over = True
                            
                            selected_pos = None
                            valid_moves = []
        
        # AI's turn
        if not env.current_player and not game_over:  # AI plays as black (False)
            state = env._get_state()
            ai_move = ai_player.get_action(state)
            if ai_move:
                # Animate and execute move
                visualizer.animate_move(*ai_move)
                state, reward, done, info = env.step(ai_move)
                
                if done:
                    game_over = True
        
        # Update display
        visualizer.draw_board(selected_pos, valid_moves)
        
        # Display game over message
        if game_over:
            winner_text = "Red Wins!" if info['winner'] else "Black Wins!" if info['winner'] is not None else "Draw!"
            text_color = visualizer.RED if info['winner'] else visualizer.BLACK if info['winner'] is not None else (128, 128, 128)
            
            font = pygame.font.Font(None, 74)
            text = font.render(winner_text, True, text_color)
            text_rect = text.get_rect(center=(visualizer.width//2, visualizer.height//2))
            visualizer.screen.blit(text, text_rect)
            pygame.display.flip()
            
            # Keep showing the final position until user closes window
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
    
    visualizer.close()

if __name__ == "__main__":
    main() 