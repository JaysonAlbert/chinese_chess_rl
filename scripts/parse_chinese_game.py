import re
from enum import Enum
import json

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
        self.red_pieces = {
            "车": "车", "马": "马", "象": "象", "士": "士",
            "将": "将", "炮": "炮", "兵": "兵"
        }
        self.black_pieces = {
            "车": "車", "马": "馬", "象": "相", "士": "仕",
            "将": "帥", "炮": "砲", "兵": "卒"
        }

    def parse_game(self, game_text):
        """Parse entire game text into moves."""
        moves = []
        lines = game_text.strip().split('\n')
        
        for line in lines:
            # Remove move numbers and spaces
            line = re.sub(r'^\s*\d+\.', '', line).strip()
            
            # Split into red and black moves
            parts = line.split()
            if len(parts) >= 2:
                moves.append(self.parse_move(parts[0], is_red=True))   # Red move
                moves.append(self.parse_move(parts[1], is_red=False))  # Black move
        
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
    with open('resources/game.txt', 'r', encoding='utf-8') as f:
        game_text = f.read()
    
    parser = ChineseNotationParser()
    moves = parser.parse_game(game_text)
    
    # Print moves for debugging
    print("Parsed moves:")
    for i, move in enumerate(moves):
        print(f"{i+1}. {move}")
    
    # Convert to JSON for easier processing
    with open('resources/parsed_game.json', 'w', encoding='utf-8') as f:
        json.dump(moves, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully parsed {len(moves)} moves and saved to parsed_game.json")

if __name__ == "__main__":
    main() 