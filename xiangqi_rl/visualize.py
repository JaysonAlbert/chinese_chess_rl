import pygame
import numpy as np
import os
import platform
import logging

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    def __init__(self, env, animation_duration=1000, debug=False):
        """
        Initialize the visualizer
        Args:
            env: The game environment
            animation_duration: Duration of move animations in milliseconds (default: 1000)
            debug: Whether to draw intersection points for debugging (default: False)
        """
        self.env = env
        self.animation_duration = animation_duration
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
        self.debug = debug
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()
    
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
            logger.warning("No suitable Chinese font found. Pieces may not display correctly.")
        
        # 对于棋子文字，我们使用稍大一点的字号以确保清晰可见
        font = pygame.font.SysFont(font_name, int(size * 0.7))  # 增大字号到原来的0.7倍
        
        try:
            text = font.render(symbol, True, color)
            text_rect = text.get_rect(center=(size//2, size//2))
            surface.blit(text, text_rect)
        except pygame.error as e:
            logger.error(f"Error rendering piece text: {e}")
            # 如果渲染失败，尝试使用备用字符
            fallback_symbols = {'r': 'R', 'h': 'H', 'e': 'E', 'a': 'A', 
                              'k': 'K', 'c': 'C', 'p': 'P'}
            fallback_text = font.render(fallback_symbols[piece_type], True, color)
            fallback_rect = fallback_text.get_rect(center=(size//2, size//2))
            surface.blit(fallback_text, fallback_rect)
        
        # Save the image
        pygame.image.save(surface, filepath)
    
    def get_board_position(self, mouse_pos):
        """Convert mouse position to board coordinates"""
        x, y = mouse_pos
        
        # Calculate the size of clickable area around each intersection
        click_area = self.square_size // 2
        
        # Calculate board boundaries
        board_left = self.margin
        board_top = self.margin
        
        # Adjust position relative to board
        rel_x = x - board_left
        rel_y = y - board_top
        
        # Calculate nearest intersection
        col = round(rel_x / self.square_size)
        row = round(rel_y / self.square_size)
        
        # Check if click is close enough to an intersection
        x_dist = abs(rel_x - col * self.square_size)
        y_dist = abs(rel_y - row * self.square_size)
        
        if (x_dist <= click_area and y_dist <= click_area and 
            0 <= row < 10 and 0 <= col < 9):
            return row, col
        
        return None

    def highlight_selected(self, pos):
        """Highlight the selected piece"""
        if pos:
            row, col = pos
            x = self.margin + col * self.square_size
            y = self.margin + row * self.square_size
            
            # Draw a semi-transparent green rectangle
            s = pygame.Surface((self.square_size - 4, self.square_size - 4), pygame.SRCALPHA)
            s.fill((0, 255, 0, 128))  # Green with alpha
            rect = s.get_rect(center=(x, y))
            self.screen.blit(s, rect)

    def highlight_valid_moves(self, valid_moves):
        """Highlight valid move destinations"""
        for _, to_pos in valid_moves:
            row, col = to_pos
            x = self.margin + col * self.square_size
            y = self.margin + row * self.square_size
            
            # Draw a semi-transparent yellow circle
            s = pygame.Surface((self.square_size - 4, self.square_size - 4), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 0, 128), (s.get_width()//2, s.get_height()//2), s.get_width()//2)
            rect = s.get_rect(center=(x, y))
            self.screen.blit(s, rect)

    def draw_board(self, selected_pos=None, valid_moves=None):
        """Draw the board with optional highlights for selected piece and valid moves"""
        self.screen.fill(self.BACKGROUND)
        
        # Calculate board area
        board_left = self.margin
        board_right = board_left + 8 * self.square_size
        board_top = self.margin
        board_bottom = board_top + 9 * self.square_size
        
        # Draw column numbers at bottom (red's perspective)
        for i in range(9):
            num = self.chinese_numbers[9 - i]  # Right to left for red
            text = self.number_font.render(num, True, self.RED)
            x = board_left + i * self.square_size
            y = board_bottom + 44
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
        
        # Black's perspective (top numbers)
        for i in range(9):
            num = str(i + 1)  # Left to right for black
            text = self.number_font.render(num, True, self.BLACK)
            x = board_left + i * self.square_size
            y = board_top - 45
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
        
        # Highlight valid moves first (so pieces appear on top)
        if valid_moves:
            self.highlight_valid_moves(valid_moves)
        
        # Draw pieces
        for i in range(10):
            for j in range(9):
                piece = self.env.board[i][j]
                if piece:
                    # Calculate exact intersection position
                    x = self.margin + j * self.square_size
                    y = self.margin + i * self.square_size
                    
                    # Draw piece centered on intersection
                    img = self.pieces_img[(piece.piece_type, piece.is_red)]
                    img_rect = img.get_rect(center=(x, y))
                    self.screen.blit(img, img_rect)
        
        # Highlight selected piece last (so it appears on top)
        if selected_pos:
            self.highlight_selected(selected_pos)
        
        # Show current player
        current_player = "红方" if self.env.current_player else "黑方"
        text = self.font.render(f"当前玩家: {current_player}", True, 
                               self.RED if self.env.current_player else self.BLACK)
        text_rect = text.get_rect(center=(self.width//2, self.height - self.margin + 25))
        self.screen.blit(text, text_rect)
        
        # Debug: draw intersection points
        if self.debug:
            for i in range(10):
                for j in range(9):
                    x = self.margin + j * self.square_size
                    y = self.margin + i * self.square_size
                    pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 2)
        
        pygame.display.flip()

    def animate_move(self, from_pos, to_pos, duration=500):
        """Animate piece movement with highlights"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.env.board[from_row][from_col]
        if not piece:
            return

        start_time = pygame.time.get_ticks()
        
        # Calculate board positions
        from_x = self.margin + from_col * self.square_size
        from_y = self.margin + from_row * self.square_size
        to_x = self.margin + to_col * self.square_size
        to_y = self.margin + to_row * self.square_size

        while pygame.time.get_ticks() - start_time < duration:
            progress = min(1.0, (pygame.time.get_ticks() - start_time) / duration)
            
            # Calculate current position
            current_x = from_x + (to_x - from_x) * progress
            current_y = from_y + (to_y - from_y) * progress
            
            # Draw board without the moving piece
            self.env.board[from_row][from_col] = None
            self.draw_board()
            self.env.board[from_row][from_col] = piece
            
            # Draw highlights
            for pos, color in [(from_pos, (0, 255, 0, 128)), (to_pos, (255, 255, 0, 128))]:
                x = self.margin + pos[1] * self.square_size
                y = self.margin + pos[0] * self.square_size
                highlight = pygame.Surface((self.square_size - 10, self.square_size - 10), pygame.SRCALPHA)
                highlight.fill(color)
                self.screen.blit(highlight, highlight.get_rect(center=(x, y)))
            
            # Draw move line
            pygame.draw.line(self.screen, (0, 150, 255), (from_x, from_y), (to_x, to_y), 2)
            
            # Draw the moving piece
            img = self.pieces_img[(piece.piece_type, piece.is_red)]
            self.screen.blit(img, img.get_rect(center=(current_x, current_y)))
            
            pygame.display.flip()
            pygame.time.wait(10)
        
        # Store last move for future reference
        self.last_move = (from_pos, to_pos)
