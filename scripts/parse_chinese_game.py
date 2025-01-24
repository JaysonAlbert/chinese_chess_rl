import re
from enum import Enum
import json
import os

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

def main():
    # Read game text from file
    parser = ChineseNotationParser()
    
    # Process each game file in the games directory
    games_dir = 'resources/games'
    for game_file in os.listdir(games_dir):
        if game_file.endswith('.txt'):
            game_path = os.path.join(games_dir, game_file)
            with open(game_path, 'r', encoding='utf-8') as f:
                game_text = f.read()
            
            try:
                moves = parser.parse_game(game_text)
                
                # Create output filename based on input filename
                output_filename = os.path.splitext(game_file)[0] + '.json'
                output_path = os.path.join('resources/parsed_games', output_filename)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save parsed moves to JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(moves, f, ensure_ascii=False, indent=2)
                
                print(f"Successfully parsed {len(moves)} moves from {game_file}")
                
            except Exception as e:
                print(f"Error processing {game_file}: {str(e)}")

if __name__ == "__main__":
    main() 