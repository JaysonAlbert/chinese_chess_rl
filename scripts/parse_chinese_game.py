import re
from enum import Enum
import json
import os
import pandas as pd
from collections import defaultdict
import sys

# Add the src directory to Python path so we can import from src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from xiangqi_rl.environment import XiangqiEnv

class Piece(Enum):
    ROOK = "车"
    HORSE = "马"
    ELEPHANT = "象"
    ADVISOR = "士"
    GENERAL = "将"
    CANNON = "炮"
    PAWN = "兵"
    # Black pieces
    B_ROOK = "車"
    B_HORSE = "馬"
    B_ELEPHANT = "相"
    B_ADVISOR = "仕"
    B_GENERAL = "帥"
    B_CANNON = "砲"
    B_PAWN = "卒"

class ChineseNotationParser:
    def __init__(self):
        # Chinese numbers mapping
        self.numbers = {
            "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9,
            "１": 1, "２": 2, "３": 3, "４": 4, "５": 5,
            "６": 6, "７": 7, "８": 8, "９": 9
        }
        
        # Movement characters
        self.moves = {
            "进": "forward",
            "退": "backward",
            "平": "horizontal"
        }

        # Position markers
        self.position_markers = {
            "前": "front",
            "后": "back",
            "中": "middle"
        }

        # Piece mappings for red and black
        self.black_pieces = {
            "车": "车", "马": "马", "象": "象", "士": "士",
            "将": "将", "炮": "炮", "兵": "兵"
        }
        self.red_pieces = {
            "车": "車", "马": "馬", "象": "相", "士": "仕",
            "将": "帥", "炮": "砲", "兵": "卒"
        }

    def parse_game(self, game_text):
        """Parse entire game text into moves."""
        moves = []
        lines = game_text.strip().split('\n')
        
        # Find the first empty line to locate the start of moves
        start_index = -1
        for i, line in enumerate(lines):
            if not line.strip():
                start_index = i + 1
                break
        
        if start_index == -1:
            raise ValueError("Could not find start of moves section")
        
        # Find the end of moves (next empty line or end of file)
        end_index = len(lines)
        for i in range(start_index, len(lines)):
            if not lines[i].strip():
                end_index = i
                break
        
        # Process only the moves section
        for line in lines[start_index:end_index]:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Split into moves (usually 4 parts: 2 move numbers + 2 actual moves)
            parts = line.split()
            # Remove move numbers and clean each part
            parts = [re.sub(r'^\s*\d+\.', '', part).strip() for part in parts]
            
            # Parse moves based on number of parts
            if len(parts) >= 1:  # At least red move
                moves.append(self.parse_move(parts[0], is_red=True))
            if len(parts) >= 2:  # Red and black moves
                moves.append(self.parse_move(parts[1], is_red=False))
            if len(parts) >= 3:  # Red, black and second red
                moves.append(self.parse_move(parts[2], is_red=True))
            if len(parts) >= 4:  # Complete pair of moves
                moves.append(self.parse_move(parts[3], is_red=False))
        return moves

    def parse_move(self, move_str, is_red):
        """Parse a single move in Chinese notation."""
        move_str = move_str.strip()
        
        # Convert piece character to correct set (simplified for red, traditional for black)
        piece_char = move_str[0]
        piece = piece_char
        if is_red:
            piece = self.red_pieces.get(piece_char, piece_char)
        else:
            piece = self.black_pieces.get(piece_char, piece_char)
        
        # Handle special position markers (前/后/中)
        if move_str[0] in self.position_markers:
            position_marker = self.position_markers[move_str[0]]
            piece = self.red_pieces[move_str[1]] if is_red else self.black_pieces[move_str[1]]
            direction = move_str[2]
            destination = move_str[3]
            return {
                "piece": piece,
                "position_marker": position_marker,
                "movement": self.moves.get(direction, direction),
                "end": self.numbers.get(destination, destination)
            }
        else:
            position = move_str[1]
            direction = move_str[2]
            destination = move_str[3]
            return {
                "piece": piece,
                "start": self.numbers.get(position, position),
                "movement": self.moves.get(direction, direction),
                "end": self.numbers.get(destination, destination)
            }

def convert_moves_to_training_format(moves, board_size=(10, 9)):
    """Convert parsed moves to (from_pos, to_pos) format"""
    training_moves = []
    env = XiangqiEnv()  # You'll need to import this from environment.py
    
    for move in moves:
        try:
            # Calculate the actual board positions from the Chinese notation
            from_pos, to_pos = env.calculate_move(move, env.current_player)
            
            if from_pos and to_pos:
                # Format move as "from_row,from_col,to_row,to_col"
                move_str = f"{from_pos[0]},{from_pos[1]},{to_pos[0]},{to_pos[1]}"
                training_moves.append(move_str)
                
                # Update the environment with the move
                env.step((from_pos, to_pos))
            
        except Exception as e:
            print(f"Error converting move: {move}, error: {e}")
            continue
            
    return training_moves

def create_training_dataset(games_dir, output_path):
    """Create a CSV dataset from parsed games"""
    parser = ChineseNotationParser()
    all_games = []
    
    # Check if games directory exists
    if not os.path.exists(games_dir):
        print(f"Error: Games directory '{games_dir}' does not exist")
        return
    
    # Process each game file
    for game_file in os.listdir(games_dir):
        if game_file.endswith('.txt'):
            game_path = os.path.join(games_dir, game_file)
            try:
                with open(game_path, 'r', encoding='utf-8') as f:
                    game_text = f.read()
                
                # Parse moves from the game
                moves = parser.parse_game(game_text)
                
                # Convert moves to training format
                training_moves = convert_moves_to_training_format(moves)
                
                if training_moves:
                    # Join moves with spaces to create a single string
                    moves_str = ' '.join(training_moves)
                    
                    # Add game metadata if available
                    metadata = extract_game_metadata(game_text)
                    
                    game_data = {
                        'game_id': os.path.splitext(game_file)[0],
                        'moves': moves_str,
                        'num_moves': len(training_moves),
                        **metadata
                    }
                    
                    all_games.append(game_data)
                    print(f"Successfully processed {game_file} - {len(training_moves)} moves")
                
            except Exception as e:
                print(f"Error processing {game_file}: {str(e)}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_games)
    
    if len(df) > 0:
        df.to_csv(output_path, index=False)
        print(f"\nCreated dataset with {len(df)} games")
        
        # Print some statistics
        print(f"\nDataset Statistics:")
        print(f"Total number of games: {len(df)}")
        print(f"Average moves per game: {df['num_moves'].mean():.1f}")
        print(f"Max moves in a game: {df['num_moves'].max()}")
        print(f"Min moves in a game: {df['num_moves'].min()}")
    else:
        print("\nNo games were successfully processed. Dataset was not created.")

def extract_game_metadata(game_text):
    """Extract metadata from game text"""
    metadata = defaultdict(str)
    
    # Look for common metadata patterns
    patterns = {
        'event': r'赛事[：:]\s*(.+)',
        'date': r'日期[：:]\s*(.+)',
        'red_player': r'红方[：:]\s*(.+)',
        'black_player': r'黑方[：:]\s*(.+)',
        'result': r'结果[：:]\s*(.+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, game_text)
        if match:
            metadata[key] = match.group(1).strip()
    
    return dict(metadata)

def main():
    # Create output directories if they don't exist
    os.makedirs('resources/parsed_games', exist_ok=True)
    os.makedirs('resources/games', exist_ok=True)
    
    # Process games and create dataset
    games_dir = 'resources/games'
    output_path = 'resources/xiangqi_games.csv'
    
    # Check if there are any game files
    game_files = [f for f in os.listdir(games_dir) if f.endswith('.txt')]
    if not game_files:
        print(f"No .txt files found in {games_dir}")
        print("Please add some Chinese chess game files in .txt format to the games directory")
        return
        
    create_training_dataset(games_dir, output_path)

if __name__ == "__main__":
    main() 