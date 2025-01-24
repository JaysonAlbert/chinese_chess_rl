import sys
import os
import platform
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from src.environment import XiangqiEnv, Piece
import json
from src.visualize import XiangqiVisualizer

def play_game():
    env = XiangqiEnv()
    visualizer = XiangqiVisualizer(env)
    selected_pos = None
    valid_moves = []
    
    running = True
    while running:
        visualizer.draw_board(selected_pos, valid_moves)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // visualizer.square_size
                row = y // visualizer.square_size
                
                if selected_pos is None:
                    piece = env.board[row][col]
                    if piece and piece.is_red == env.current_player:
                        selected_pos = (row, col)
                        valid_moves = [m for m in env.get_valid_moves() if m[0] == (row, col)]
                else:
                    move = (selected_pos, (row, col))
                    if move in valid_moves:
                        _, reward, done = env.step(move)
                        if done:
                            winner = "Red" if reward > 0 else "Black"
                            print(f"Game Over! {winner} wins!")
                            pygame.time.wait(2000)
                            running = False
                    selected_pos = None
                    valid_moves = []
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    selected_pos = None
                    valid_moves = []
                elif event.key == pygame.K_r:  # Reset game
                    env.reset()
                    selected_pos = None
                    valid_moves = []
    
    pygame.quit()

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

def piece_to_type(piece):
    """Convert Chinese piece character to internal piece type"""
    mapping = {
        "车": 'r', "車": 'r',  # Rook
        "马": 'h', "馬": 'h',  # Horse
        "象": 'e', "相": 'e',  # Elephant
        "士": 'a', "仕": 'a',  # Advisor
        "将": 'k', "帥": 'k',  # King
        "炮": 'c', "砲": 'c',  # Cannon
        "兵": 'p', "卒": 'p',  # Pawn
    }
    return mapping.get(piece, '')

def calculate_end_position(from_pos, movement, destination, env):
    """Calculate destination position based on movement type"""
    if not from_pos:
        return None
        
    try:
        row, col = from_pos
        piece = env.board[row][col]
        is_red = piece.is_red
        steps = int(destination) if not isinstance(destination, int) else destination
        
        # 使用 environment.py 中的移动逻辑
        piece_type = piece.piece_type
        valid_moves = env.get_piece_moves(row, col)
        
        # 根据移动类型和目标位置找到匹配的移动
        for move in valid_moves:
            _, (target_row, target_col) = move
            
            if movement == "horizontal":
                # 水平移动到指定列
                # 红方：从右到左数（9-col），如"车二平五"中的5
                # 黑方：从左到右数（col+1），如"车2平5"中的5
                target_col_num = target_col + 1 if not is_red else 9 - target_col
                if target_col_num == steps and target_row == row:
                    return (target_row, target_col)
                    
            elif movement == "forward":
                # 前进指定步数
                # 红方：向上移动（行号减小），如"兵七进一"
                # 黑方：向下移动（行号增大），如"卒7进1"
                steps_moved = abs(target_row - row)
                if steps_moved == steps:
                    # 确认移动方向正确
                    if (is_red and target_row < row) or (not is_red and target_row > row):
                        # 确认是直线移动
                        if target_col == col:
                            return (target_row, target_col)
                            
            elif movement == "backward":
                # 后退指定步数
                # 红方：向下移动（行号增大），如"兵七退一"
                # 黑方：向上移动（行号减小），如"卒7退1"
                steps_moved = abs(target_row - row)
                if steps_moved == steps:
                    # 确认移动方向正确
                    if (is_red and target_row > row) or (not is_red and target_row < row):
                        # 确认是直线移动
                        if target_col == col:
                            return (target_row, target_col)
                            
            elif piece_type == 'h':  # 马的特殊移动
                if movement == "forward":
                    # 马前进
                    # 红方：向上移动（行号减小），如"马二进三"
                    # 黑方：向下移动（行号增大），如"马2进3"
                    if ((is_red and target_row < row) or (not is_red and target_row > row)):
                        col_diff = target_col - col
                        # 7表示进右，9表示进左
                        if (steps == 7 and col_diff == 1) or (steps == 9 and col_diff == -1):
                            return (target_row, target_col)
                elif movement == "backward":
                    # 马后退
                    # 红方：向下移动（行号增大），如"马二退三"
                    # 黑方：向上移动（行号减小），如"马2退3"
                    if ((is_red and target_row > row) or (not is_red and target_row < row)):
                        col_diff = target_col - col
                        # 7表示退右，9表示退左
                        if (steps == 7 and col_diff == 1) or (steps == 9 and col_diff == -1):
                            return (target_row, target_col)
        
    except (TypeError, ValueError) as e:
        print(f"Error calculating end position: {e}")
    
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, help='Path to the recorded game JSON file')
    args = parser.parse_args()
    
    if args.game:
        play_recorded_game(args.game)
    else:
        play_game() 