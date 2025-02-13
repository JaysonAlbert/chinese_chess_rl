import sys
import os
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import cProfile
import pstats
from datetime import datetime

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
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
        # Initialize agent with MCTS search and pass the environment
        self.agent = XiangqiAgent(self.model, self.env, num_simulations=10)

    def get_action(self):
        """Get the AI's move based on the current state"""
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return None
            
        # Use the agent to select an action with MCTS
        return self.agent.select_action(valid_moves)  # Lower temperature for stronger play

def find_model():
    """Find a valid model in the checkpoints directory if no model path is provided"""
    checkpoint_dir = os.path.join(project_root, 'logs', 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        return None
        
    # Try different model file patterns
    patterns = [
        'final_model.pt',
        'model_iteration_*.pt',
        'model_iteration_*.pth',
        'best_model.pt',
        'best_model.pth',
        'final_model.pth'
    ]
    
    for pattern in patterns:
        import glob
        model_files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if model_files:
            return model_files[0]  # Return the first matching model file
            
    return None

def show_loading_screen(visualizer, message):
    """Display a loading message on the game board"""
    visualizer.screen.fill(visualizer.BACKGROUND)
    font = pygame.font.Font(None, 36)
    text = font.render(message, True, (0, 0, 0))
    text_rect = text.get_rect(center=(visualizer.width//2, visualizer.height//2))
    visualizer.screen.blit(text, text_rect)
    pygame.display.flip()

def run_with_profiler(func):
    """Run the given function with profiler and save results"""
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = os.path.join(project_root, 'logs', 'profiles', f'profile_{timestamp}.prof')
    
    # Ensure profiles directory exists
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    
    # Run the profiler
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        func()
    finally:
        profiler.disable()
        
    # Save and print the stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Save detailed stats to file
    stats.dump_stats(profile_path)
    
    # Print top 20 time-consuming functions
    print("\nTop 20 time-consuming functions:")
    stats.strip_dirs()
    stats.sort_stats('cumulative').print_stats(20)
    
    print(f"\nDetailed profile saved to: {profile_path}")
    print("To analyze the profile, you can use:")
    print(f"python -m pstats {profile_path}")
    print("or")
    print(f"snakeviz {profile_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Play Xiangqi against AI or watch AI vs AI')
    parser.add_argument('--model', type=str, help='Path to the model file for first AI', default=None)
    parser.add_argument('--model2', type=str, help='Path to the model file for second AI (optional)', default=None)
    parser.add_argument('--mode', type=str, choices=['human', 'ai'], default='human',
                       help='Game mode: human vs AI or AI vs AI')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                       help='Device for first AI (default: cuda if available, else cpu)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    args = parser.parse_args()
    
    # Store args in global scope for the wrapped main function
    global ARGS
    ARGS = args
    
    def wrapped_main():
        # Initialize environment and visualizer
        env = XiangqiEnv()
        visualizer = XiangqiVisualizer(env, debug=False)
        
        # Show initial loading screen
        show_loading_screen(visualizer, "Loading AI models...")
        
        # Determine device
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device1 = args.device if args.device else default_device
        
        # Use provided model paths or find one
        model_path = args.model if args.model else find_model()
        if model_path:
            print(f"Using model for first AI at: {model_path} (device: {device1})")
        else:
            print(f"No trained model found for first AI - it will play randomly (device: {device1})")
        
        # Initialize first AI player (plays as Black)
        show_loading_screen(visualizer, "Initializing Black AI...")
        ai_black = AIPlayer(env, model_path, device=device1)
        
        # Initialize second AI player if in AI vs AI mode
        ai_red = None
        if args.mode == 'ai':
            model2_path = args.model2 if args.model2 else model_path
            if model2_path == model_path:
                show_loading_screen(visualizer, "Initializing Red AI (using same model)...")
                # Reuse the same AI instance if using the same model
                print(f"Using same model for second AI (device: {device1})")
                ai_red = AIPlayer(env, None, device=device1)
                # Share the model and agent from first AI
                ai_red.model = ai_black.model
                ai_red.agent = ai_black.agent
            else:
                show_loading_screen(visualizer, "Initializing Red AI...")
                print(f"Using model for second AI at: {model2_path} (device: {device1})")
                ai_red = AIPlayer(env, model2_path, device=device1)
            print("\nAI vs AI mode:")
            print("- Red AI (bottom) vs Black AI (top)")
        else:
            print("\nGame Controls:")
            print("- Click on a piece to select it")
            print("- Click on a valid destination to move")
            print("- You play as Red (bottom)")
            print("- AI plays as Black (top)")
        print("- Close window to quit\n")
        
        # Draw initial board state
        visualizer.draw_board(None, [])
        pygame.display.flip()
        
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
                    # Handle human player's move (only in human vs AI mode)
                    if args.mode == 'human' and env.current_player:  # Human plays as red (True)
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
                                    state = env.step((selected_pos, board_pos))
                                    
                                    if env.is_game_over:
                                        game_over = True
                                
                                selected_pos = None
                                valid_moves = []
            
            # AI's turn
            if not game_over:
                current_ai = None
                if env.current_player:  # Red's turn
                    if args.mode == 'ai':
                        current_ai = ai_red
                else:  # Black's turn
                    current_ai = ai_black
                
                if current_ai:
                    # Update display and handle events before AI move
                    visualizer.draw_board(selected_pos, valid_moves)
                    pygame.display.flip()
                    
                    # Handle any pending events to keep window responsive
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                    
                    if not running:
                        break
                    
                    ai_move = current_ai.get_action()
                    if ai_move:
                        # Animate and execute move
                        visualizer.animate_move(*ai_move)
                        state = env.step(ai_move)
                        
                        if env.is_game_over:
                            game_over = True
                        
                        # Add a small delay between AI moves in AI vs AI mode
                        if args.mode == 'ai':
                            # Update display before delay
                            visualizer.draw_board(selected_pos, valid_moves)
                            pygame.display.flip()
                            pygame.time.wait(500)  # 500ms delay
            
            # Update display
            visualizer.draw_board(selected_pos, valid_moves)
            
            # Display game over message
            if game_over:
                winner_text = "Red Wins!" if env.winner else "Black Wins!" if env.winner is not None else "Draw!"
                text_color = visualizer.RED if env.winner else visualizer.BLACK if env.winner is not None else (128, 128, 128)
                
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
    
    # Run with or without profiler based on args
    if args.profile:
        run_with_profiler(wrapped_main)
    else:
        wrapped_main()

if __name__ == "__main__":
    main() 