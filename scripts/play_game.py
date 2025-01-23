import sys
import os
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
        self.font = pygame.font.SysFont("wenquanyimicrohei", 36)
        self.number_font = pygame.font.SysFont("wenquanyimicrohei", 24)  # Smaller font for numbers
        self.river_font = pygame.font.SysFont("wenquanyimicrohei", 40)  # Larger font for river text
        
        # Add Chinese number mapping
        self.chinese_numbers = {
            1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
            6: "六", 7: "七", 8: "八", 9: "九"
        }
    
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
        font = pygame.font.SysFont("wenquanyimicrohei", size//2)
        text = font.render(symbol, True, color)
        text_rect = text.get_rect(center=(size//2, size//2))
        surface.blit(text, text_rect)
        
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
        
        pygame.display.flip()

def load_chinese_game(json_file):
    """Load and convert Chinese notation moves to engine format."""
    with open(json_file, 'r', encoding='utf-8') as f:
        moves = json.load(f)
    
    converted_moves = []
    for move in moves:
        # Convert the Chinese notation to engine format
        # This will need to be adapted based on your engine's move format
        converted_move = convert_chinese_to_engine_format(move)
        converted_moves.append(converted_move)
    
    return converted_moves

def convert_chinese_to_engine_format(move):
    """Convert a Chinese notation move to engine format."""
    # This function needs to be implemented based on your engine's move format
    # Example conversion:
    piece = move['piece']
    start = move['start']
    movement = move['movement']
    end = move['end']
    
    # Convert to your engine's coordinate system
    # This is a placeholder - implement based on your engine's requirements
    return {
        'from': calculate_position(piece, start, movement, env),
        'to': calculate_end_position(calculate_position(piece, start, movement, env), movement, end, env)
    }

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
        
        # Play the next move if available
        if move_index < len(recorded_moves) and current_time - last_move_time >= 1000:
            move = recorded_moves[move_index]
            is_red_move = (move_index % 2 == 0)
            print(f"\nProcessing move {move_index + 1}: {move} ({'Red' if is_red_move else 'Black'} to move)")
            
            # Verify current player
            if env.current_player != is_red_move:
                print(f"Warning: Expected {'Red' if is_red_move else 'Black'} to move")
                env.current_player = is_red_move
            
            # Calculate and make move
            from_pos, to_pos = env.calculate_move(move)
            
            if from_pos and to_pos:
                print(f"Making move from {from_pos} to {to_pos}")
                if (from_pos, to_pos) in env.get_valid_moves():
                    _, reward, done = env.step((from_pos, to_pos))
                    print(f"Move made, reward: {reward}, done: {done}")
                else:
                    print("Move is not valid according to game rules")
            else:
                print("Failed to calculate valid positions for this move")
            
            move_index += 1
            last_move_time = current_time
        
        # Update display
        visualizer.draw_board(env)
        pygame.display.flip()
        pygame.time.Clock().tick(60)
    
    pygame.quit()

def calculate_position(piece, pos, movement, env):
    """Calculate board position from Chinese notation"""
    # Handle special position markers (前/后)
    if isinstance(pos, dict) and 'position_marker' in pos:
        # Find all pieces of this type
        pieces_pos = []
        for i in range(10):
            for j in range(9):
                current_piece = env.board[i][j]
                if current_piece and current_piece.piece_type == piece_to_type(piece):
                    if (current_piece.is_red and piece in "车马象士将炮兵") or \
                       (not current_piece.is_red and piece in "車馬相仕帥砲卒"):
                        pieces_pos.append((i, j))
        
        # Sort positions based on the marker
        if pos['position_marker'] == 'front':
            # For red pieces, "front" means larger row number (closer to black)
            # For black pieces, "front" means smaller row number (closer to red)
            is_red = piece in "车马象士将炮兵"
            pieces_pos.sort(key=lambda x: x[0], reverse=is_red)
            return pieces_pos[0] if pieces_pos else None
        elif pos['position_marker'] == 'back':
            is_red = piece in "车马象士将炮兵"
            pieces_pos.sort(key=lambda x: x[0], reverse=not is_red)
            return pieces_pos[0] if pieces_pos else None
    else:
        try:
            is_red = piece in "车马象士将炮兵"
            # Convert file number (1-9) to board column (0-8)
            # For red pieces: count from right to left (9 - pos)
            # For black pieces: count from left to right (pos - 1)
            col = (9 - int(pos)) if is_red else (int(pos) - 1)
            
            piece_type = piece_to_type(piece)
            # Search for the piece in the correct starting area
            if is_red:
                # Red pieces are at the bottom, search from bottom up
                rows = range(9, -1, -1)
            else:
                # Black pieces are at the top, search from top down
                rows = range(10)
            
            for row in rows:
                current_piece = env.board[row][col]
                if current_piece and current_piece.piece_type == piece_type:
                    if (current_piece.is_red and is_red) or \
                       (not current_piece.is_red and not is_red):
                        return (row, col)
        except (TypeError, ValueError) as e:
            print(f"Error processing position {pos}: {e}")
    return None

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
        
        if piece.piece_type == 'h':  # Horse moves
            if movement == "forward":
                # Move forward 2, then 1 right/left
                new_row = row + (-2 if is_red else 2)  # 2 steps forward
                if steps == 7:  # forward-right
                    new_col = col + 1
                else:  # steps == 9, forward-left
                    new_col = col - 1
                return (new_row, new_col)
            elif movement == "backward":
                # Move backward 2, then 1 right/left
                new_row = row + (2 if is_red else -2)  # 2 steps backward
                if steps == 7:  # backward-right
                    new_col = col + 1
                else:  # steps == 9, backward-left
                    new_col = col - 1
                return (new_row, new_col)
        
        else:  # Normal piece movements
            if movement == "horizontal":
                # For horizontal moves, destination is the target column (1-9)
                new_col = (9 - steps) if is_red else (steps - 1)
                return (row, new_col)
            
            elif movement == "forward":
                # For forward moves, steps indicates number of squares to move
                if is_red:
                    new_row = row - steps
                else:
                    new_row = row + steps
                return (new_row, col)
            
            elif movement == "backward":
                # For backward moves, steps indicates number of squares to move
                if is_red:
                    new_row = row + steps
                else:
                    new_row = row - steps
                return (new_row, col)
            
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
        play_game()  # Original interactive mode 