import pygame
import numpy as np
import os
import platform


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
        
class XiangqiVisualizer:
    def __init__(self, env):
        self.env = env
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
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
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
    
    def draw_board(self, selected_pos=None, valid_moves=None):
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
                piece = self.env.board[i][j]
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
        current_player = "红方" if self.env.current_player else "黑方"
        text = self.font.render(f"当前玩家: {current_player}", True, self.RED if self.env.current_player else self.BLACK)
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
