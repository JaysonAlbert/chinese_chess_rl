import sys
import os
import platform
import time
import pandas as pd
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from xiangqi_rl.environment import XiangqiEnv, Piece
import json
from xiangqi_rl.visualize import XiangqiVisualizer
import logging

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_games(csv_path):
    """Load games from CSV file"""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} games from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None

def play_game(moves_str, delay=1.0, visualize=True):
    """Play through a single game"""
    env = XiangqiEnv()
    vis = None
    if visualize:
        vis = XiangqiVisualizer(env)
        pygame.init()
        pygame.display.set_mode((vis.width, vis.height), pygame.NOFRAME | pygame.SHOWN)
        pygame.display.flip()
    
    moves = moves_str.split()
    logger.info(f"Playing game with {len(moves)} moves...")
    
    try:
        for i, move_str in enumerate(moves, 1):
            # Handle pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if visualize:
                        vis.close()
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if visualize:
                        vis.close()
                    return False

            from_row, from_col, to_row, to_col = map(int, move_str.split(','))
            from_pos = (from_row, from_col)
            to_pos = (to_row, to_col)
            
            # Get piece info and make move
            piece = env.board[from_row][from_col]
            if piece:
                logger.info(f"Move {i}: {'Red' if piece.is_red else 'Black'} {piece.piece_type} from {from_pos} to {to_pos}")
            
            if visualize:
                # Draw initial state and animate
                vis.draw_board()
                vis.animate_move(from_pos, to_pos)
                pygame.display.flip()
            
            # Update game state after animation
            state, reward, done, info = env.step((from_pos, to_pos))
            
            if visualize:
                # Draw final position and wait
                vis.draw_board()
                pygame.display.flip()
                pygame.time.wait(int(delay * 1000))
            
            if done:
                logger.info(f"Game over after {i} moves!")
                logger.info(f"Winner: {'Red' if env.winner else 'Black'}" if env.winner is not None else "Draw")
                if visualize:
                    pygame.time.wait(2000)
                break
        
        if visualize:
            vis.close()
        return True
            
    except Exception as e:
        logger.error(f"Error playing move: {e}")
        if visualize:
            vis.close()
        return False

def main():
    # Load games from CSV
    csv_path = 'resources/xiangqi_games.csv'
    df = load_games(csv_path)
    if df is None:
        return
    
    # Ask user which game to play
    while True:
        print("\nAvailable games:")
        for i, (game_id, moves) in enumerate(zip(df['game_id'], df['moves']), 1):
            num_moves = len(moves.split())
            print(f"{i}. Game {game_id} ({num_moves} moves)")
        
        try:
            choice = input("\nEnter game number to play (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            game_idx = int(choice) - 1
            if 0 <= game_idx < len(df):
                # Get game details
                game = df.iloc[game_idx]
                print(f"\nPlaying game {game['game_id']}")
                if 'event' in game and game['event']:
                    print(f"Event: {game['event']}")
                if 'red_player' in game and game['red_player']:
                    print(f"Red: {game['red_player']}")
                if 'black_player' in game and game['black_player']:
                    print(f"Black: {game['black_player']}")
                
                # Ask for delay between moves
                try:
                    delay = float(input("Enter delay between moves (seconds, default 1.0): ") or 1.0)
                except ValueError:
                    delay = 1.0
                
                # Play the game
                success = play_game(game['moves'], delay=delay)
                if not success:
                    print("Game playback interrupted")
                    break
            else:
                print("Invalid game number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nPlayback interrupted by user")
            break

def play_recorded_game(game_file):
    """Play a recorded game from a JSON file"""
    env = XiangqiEnv()
    visualizer = XiangqiVisualizer(env)
    
    # Load moves from JSON
    print(f"Loading moves from {game_file}")
    with open(game_file, 'r', encoding='utf-8') as f:
        recorded_moves = json.load(f)
    print(f"Loaded {len(recorded_moves)} moves")
    
    running = True
    move_index = 0
    last_move_time = 0
    waiting_for_next_move = False
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Reset game
                    env.reset()
                    move_index = 0
                    last_move_time = current_time
                    waiting_for_next_move = False
        
        # 只有在不处于等待状态时才处理下一步移动
        if move_index < len(recorded_moves) and not waiting_for_next_move:
            if current_time - last_move_time >= visualizer.MOVE_INTERVAL:
                move = recorded_moves[move_index]
                is_red_move = (move_index % 2 == 0)
                print(f"\nProcessing move {move_index + 1}: {move} ({'Red' if is_red_move else 'Black'} to move)")
                
                # Verify current player
                if env.current_player != is_red_move:
                    print(f"Warning: Expected {'Red' if is_red_move else 'Black'} to move")
                    env.current_player = is_red_move
                
                # Calculate and make move using environment's logic
                from_pos, to_pos = env.calculate_move(move, is_red_move)
                
                if from_pos and to_pos:
                    print(f"Making move from {from_pos} to {to_pos}")
                    piece = env.board[from_pos[0]][from_pos[1]]
                    if piece:
                        valid_moves = piece.get_moves(from_pos, env.board)
                        if (from_pos, to_pos) in valid_moves:
                            visualizer.last_piece_type = piece.piece_type
                            visualizer.last_piece_is_red = piece.is_red
                            visualizer.last_move = (from_pos, to_pos)
                            visualizer.move_start_time = current_time
                            
                            visualizer.animate_move(from_pos, to_pos)
                            _, reward, done = env.step((from_pos, to_pos))
                            move_index += 1
                            waiting_for_next_move = True  # 设置等待状态
                            last_move_time = current_time  # 更新最后移动时间
                            
                            if done:
                                winner = "红方" if reward > 0 else "黑方"
                                print(f"游戏结束！{winner}获胜！")
                                pygame.time.wait(3000)
                                running = False
                        else:
                            # 打印错误信息和合法移动
                            error_msg = [
                                f"错误：第{move_index + 1}步移动不合法！",
                                f"移动：{move}",
                                f"位置：从{from_pos}到{to_pos}",
                                "",
                                "该棋子的所有合法移动："
                            ]
                            if valid_moves:
                                for i, (_, valid_to) in enumerate(valid_moves, 1):
                                    error_msg.append(f"{i}. 到 {valid_to}")
                            else:
                                error_msg.append("该棋子当前没有合法移动！")
                            
                            print("\n".join(error_msg))
                            
                            # 等待用户按键继续
                            waiting = True
                            while waiting:
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        waiting = False
                                        running = False
                                    elif event.type == pygame.KEYDOWN:
                                        if event.key in [pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN]:
                                            waiting = False
                                pygame.time.wait(100)  # 减少CPU使用
                            
                            running = False
                    else:
                        print(f"错误：第{move_index + 1}步起始位置没有棋子！")
                        print(f"移动：{move}")
                        print(f"位置：从{from_pos}到{to_pos}")
                        pygame.time.wait(3000)
                        running = False
                else:
                    print(f"错误：第{move_index + 1}步无法解析！")
                    print(f"移动：{move}")
                    print(f"位置：从{from_pos}到{to_pos}")
                    pygame.time.wait(3000)
                    running = False
        
        # 检查是否可以结束等待状态
        if waiting_for_next_move and current_time - last_move_time >= visualizer.MOVE_DURATION:
            waiting_for_next_move = False
        
        visualizer.draw_board()
        pygame.display.flip()
        pygame.time.Clock().tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        pygame.quit() 