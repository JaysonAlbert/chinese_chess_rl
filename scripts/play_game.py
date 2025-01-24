import sys
import os
import platform
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from src.environment import XiangqiEnv, Piece
import json

class ChessVisualizer:
    def __init__(self):
        pygame.init()
        self.square_size = 60
        self.margin = 60  # Increase margin for better visibility of numbers
        self.bottom_margin = 60  # Bottom margin for player text
        # Width should be 8 spaces (9 lines)
        self.width = 8 * self.square_size + 2 * self.margin
        # Height should be 9 spaces (10 lines) plus extra space for numbers
        self.height = 9 * self.square_size + 2 * self.margin + self.bottom_margin
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Chinese Chess')
        
        # Colors
        self.BACKGROUND = (255, 223, 162)
        self.LINES = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        
        # Load piece images
        self.pieces_img = {}
        self.load_pieces()
        
        # Font for UI elements
        self.font = pygame.font.SysFont(get_system_font(), 36)
        self.number_font = pygame.font.SysFont(get_system_font(), 24)  # Smaller font for numbers
        self.river_font = pygame.font.SysFont(get_system_font(), 40)  # Larger font for river text
        
        # Add Chinese number mapping
        self.chinese_numbers = {
            1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
            6: "六", 7: "七", 8: "八", 9: "九"
        }
        
        # 添加新的颜色定义
        self.HIGHLIGHT_FROM = (0, 255, 0, 128)  # 起点高亮（半透明绿色）
        self.HIGHLIGHT_TO = (255, 255, 0, 128)  # 终点高亮（半透明黄色）
        self.MOVE_LINE = (0, 150, 255)  # 移动轨迹线（蓝色）
        
        # 修改动画相关属性
        self.last_move = None
        self.move_start_time = 0
        self.MOVE_DURATION = 1000  # 增加到1秒 (原来是500毫秒)
        self.MOVE_INTERVAL = 1000  # 每步移动之间的间隔时间(1秒)
    
    def load_pieces(self):
        """Load piece images from the assets directory"""
        # Create assets directory if it doesn't exist
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(assets_dir, exist_ok=True)
        
        # Define piece image filenames
        piece_files = {
            ('r', True): 'red_chariot.png',
            ('h', True): 'red_horse.png',
            ('e', True): 'red_elephant.png',
            ('a', True): 'red_advisor.png',
            ('k', True): 'red_general.png',
            ('c', True): 'red_cannon.png',
            ('p', True): 'red_pawn.png',
            ('r', False): 'black_chariot.png',
            ('h', False): 'black_horse.png',
            ('e', False): 'black_elephant.png',
            ('a', False): 'black_advisor.png',
            ('k', False): 'black_general.png',
            ('c', False): 'black_cannon.png',
            ('p', False): 'black_pawn.png',
        }
        
        # Create default piece images if they don't exist
        for (piece_type, is_red), filename in piece_files.items():
            filepath = os.path.join(assets_dir, filename)
            if not os.path.exists(filepath):
                self.create_default_piece_image(piece_type, is_red, filepath)
            
            # Load and scale the image
            img = pygame.image.load(filepath)
            img = pygame.transform.scale(img, (self.square_size - 10, self.square_size - 10))
            self.pieces_img[(piece_type, is_red)] = img
    
    def create_default_piece_image(self, piece_type, is_red, filepath):
        """Create a default piece image if the image file doesn't exist"""
        size = self.square_size - 10
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw circle background
        pygame.draw.circle(surface, (255, 255, 255), (size//2, size//2), size//2)
        pygame.draw.circle(surface, (0, 0, 0), (size//2, size//2), size//2, 2)
        
        # Chinese characters for pieces
        red_pieces = {
            'r': '車', 'h': '馬', 'e': '相', 'a': '仕', 
            'k': '帥', 'c': '炮', 'p': '兵'
        }
        black_pieces = {
            'r': '車', 'h': '馬', 'e': '象', 'a': '士',
            'k': '將', 'c': '砲', 'p': '卒'
        }
        
        # Draw piece symbol
        color = self.RED if is_red else self.BLACK
        symbol = red_pieces[piece_type] if is_red else black_pieces[piece_type]
        
        # 使用系统中文字体
        font_name = get_system_font()
        if not font_name:  # 如果没有找到合适的字体，打印警告
            print("Warning: No suitable Chinese font found. Pieces may not display correctly.")
        
        # 对于棋子文字，我们使用稍大一点的字号以确保清晰可见
        font = pygame.font.SysFont(font_name, int(size * 0.7))  # 增大字号到原来的0.7倍
        
        try:
            text = font.render(symbol, True, color)
            text_rect = text.get_rect(center=(size//2, size//2))
            surface.blit(text, text_rect)
        except pygame.error as e:
            print(f"Error rendering piece text: {e}")
            # 如果渲染失败，尝试使用备用字符
            fallback_symbols = {'r': 'R', 'h': 'H', 'e': 'E', 'a': 'A', 
                              'k': 'K', 'c': 'C', 'p': 'P'}
            fallback_text = font.render(fallback_symbols[piece_type], True, color)
            fallback_rect = fallback_text.get_rect(center=(size//2, size//2))
            surface.blit(fallback_text, fallback_rect)
        
        # Save the image
        pygame.image.save(surface, filepath)
    
    def draw_board(self, env, selected_pos=None, valid_moves=None):
        self.screen.fill(self.BACKGROUND)
        
        # Calculate board area with extra space for numbers
        board_left = self.margin
        board_right = board_left + 8 * self.square_size
        board_top = self.margin
        board_bottom = board_top + 9 * self.square_size
        
        # Draw column numbers at bottom (red's perspective)
        for i in range(9):
            num = self.chinese_numbers[9 - i]  # Right to left for red
            text = self.number_font.render(num, True, self.RED)
            x = board_left + i * self.square_size  # Align with vertical lines
            y = board_bottom + 44  # Move numbers down a bit
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
        
        # Black's perspective (top numbers)
        for i in range(9):
            num = str(i + 1)  # Left to right for black
            text = self.number_font.render(num, True, self.BLACK)
            x = board_left + i * self.square_size  # Align with vertical lines
            y = board_top - 45  # Move numbers up a bit
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
        
        # Draw horizontal lines (all of them)
        for i in range(10):  # 10 horizontal lines
            y = board_top + i * self.square_size
            pygame.draw.line(self.screen, self.LINES,
                            (board_left, y),
                            (board_right, y))
        
        # Draw vertical lines, with a gap in the middle for river (except edges)
        for i in range(9):  # 9 vertical lines
            x = board_left + i * self.square_size
            if i == 0 or i == 8:  # Leftmost and rightmost lines
                # Draw full line
                pygame.draw.line(self.screen, self.LINES,
                               (x, board_top),
                               (x, board_bottom))
            else:
                # Draw top half
                pygame.draw.line(self.screen, self.LINES,
                               (x, board_top),
                               (x, board_top + 4 * self.square_size))
                # Draw bottom half
                pygame.draw.line(self.screen, self.LINES,
                               (x, board_top + 5 * self.square_size),
                               (x, board_bottom))
        
        # Draw "楚河汉界" in the middle
        river_y = self.margin + 4.5 * self.square_size  # Middle of the board
        # Draw "楚河" on the left side
        text = self.river_font.render("楚河", True, self.BLACK)
        text_rect = text.get_rect(center=(2.5 * self.square_size + self.margin, river_y))
        self.screen.blit(text, text_rect)
        # Draw "汉界" on the right side
        text = self.river_font.render("汉界", True, self.BLACK)
        text_rect = text.get_rect(center=(6.5 * self.square_size + self.margin, river_y))
        self.screen.blit(text, text_rect)
        
        # Draw pieces on intersections
        for i in range(10):
            for j in range(9):
                piece = env.board[i][j]
                if piece:
                    # Calculate intersection position
                    x = board_left + j * self.square_size  # Align with vertical lines
                    y = board_top + i * self.square_size   # Align with horizontal lines
                    
                    img = self.pieces_img[(piece.piece_type, piece.is_red)]
                    img_rect = img.get_rect(center=(x, y))  # Center on intersection
                    self.screen.blit(img, img_rect)
        
        # Highlight selected piece (also adjust for intersection)
        if selected_pos is not None:
            x = board_left + selected_pos[1] * self.square_size
            y = board_top + selected_pos[0] * self.square_size
            highlight_size = self.square_size - 10
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (x - highlight_size//2, y - highlight_size//2,
                            highlight_size, highlight_size), 2)
        
        # Highlight valid moves (also adjust for intersection)
        if valid_moves:
            for _, to_pos in valid_moves:
                x = board_left + to_pos[1] * self.square_size
                y = board_top + to_pos[0] * self.square_size
                s = pygame.Surface((self.square_size - 10, self.square_size - 10))
                s.set_alpha(128)
                s.fill((0, 255, 0))
                s_rect = s.get_rect(center=(x, y))
                self.screen.blit(s, s_rect)
        
        # Show current player text lower but still within screen
        current_player = "红方" if env.current_player else "黑方"
        text = self.font.render(f"当前玩家: {current_player}", True, self.RED if env.current_player else self.BLACK)
        text_rect = text.get_rect(center=(self.width//2, self.height - self.margin + 25))  # Position closer to bottom
        self.screen.blit(text, text_rect)
        
        # 绘制最后一步移动的效果
        if self.last_move:
            from_pos, to_pos = self.last_move
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.move_start_time
            
            if elapsed < self.MOVE_DURATION:
                # 绘制移动动画
                progress = elapsed / self.MOVE_DURATION
                self._draw_move_animation(from_pos, to_pos, progress)
            else:
                # 动画结束后显示静态效果
                self._draw_move_effects(from_pos, to_pos)
        
        pygame.display.flip()

    def _draw_move_effects(self, from_pos, to_pos):
        """绘制移动效果（起点、终点高亮和轨迹线）"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 计算棋盘上的实际坐标
        from_x = self.margin + from_col * self.square_size
        from_y = self.margin + from_row * self.square_size
        to_x = self.margin + to_col * self.square_size
        to_y = self.margin + to_row * self.square_size
        
        # 绘制起点高亮
        highlight = pygame.Surface((self.square_size - 10, self.square_size - 10), pygame.SRCALPHA)
        highlight.fill(self.HIGHLIGHT_FROM)
        highlight_rect = highlight.get_rect(center=(from_x, from_y))
        self.screen.blit(highlight, highlight_rect)
        
        # 绘制终点高亮
        highlight = pygame.Surface((self.square_size - 10, self.square_size - 10), pygame.SRCALPHA)
        highlight.fill(self.HIGHLIGHT_TO)
        highlight_rect = highlight.get_rect(center=(to_x, to_y))
        self.screen.blit(highlight, highlight_rect)
        
        # 绘制移动轨迹线
        pygame.draw.line(self.screen, self.MOVE_LINE, (from_x, from_y), (to_x, to_y), 2)

    def _draw_move_animation(self, from_pos, to_pos, progress):
        """绘制移动动画"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 计算起点和终点的实际坐标
        from_x = self.margin + from_col * self.square_size
        from_y = self.margin + from_row * self.square_size
        to_x = self.margin + to_col * self.square_size
        to_y = self.margin + to_row * self.square_size
        
        # 计算当前位置
        current_x = from_x + (to_x - from_x) * progress
        current_y = from_y + (to_y - from_y) * progress
        
        # 绘制移动轨迹
        pygame.draw.line(self.screen, self.MOVE_LINE, (from_x, from_y), (to_x, to_y), 2)
        
        # 绘制起点和终点高亮
        self._draw_move_effects(from_pos, to_pos)
        
        # 绘制移动中的棋子
        piece = self.pieces_img[(self.last_piece_type, self.last_piece_is_red)]
        piece_rect = piece.get_rect(center=(current_x, current_y))
        self.screen.blit(piece, piece_rect)

def play_game():
    env = XiangqiEnv()
    visualizer = ChessVisualizer()
    selected_pos = None
    valid_moves = []
    
    running = True
    while running:
        visualizer.draw_board(env, selected_pos, valid_moves)
        
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
    visualizer = ChessVisualizer()
    
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
        
        visualizer.draw_board(env)
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

def get_system_font():
    system = platform.system()
    if system == 'Windows':
        fonts = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',          # 中文黑体
            'SimSun',          # 中文宋体
            'KaiTi',           # 楷体
            'NSimSun',         # 新宋体
            'arial',
            'helvetica'
        ]
    elif system == 'Linux':
        fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # Google Noto字体
            'Noto Sans SC',
            'DejaVuSans',
            'FreeSans',
            'Liberation Sans'
        ]
    else:  # MacOS
        fonts = [
            'PingFang SC',      # 苹方
            'STHeiti',          # 华文黑体
            'Hiragino Sans GB', # 冬青黑体
            'Microsoft YaHei',  # 微软雅黑
            'Arial Unicode MS',
            'FreeSans',
            'Arial'
        ]
    
    available_fonts = [f.lower() for f in pygame.font.get_fonts()]
    for font in fonts:
        if font.lower().replace(' ', '') in available_fonts:
            return font
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