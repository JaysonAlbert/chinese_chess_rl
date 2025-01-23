import numpy as np

class Piece:
    def __init__(self, piece_type, is_red):
        self.piece_type = piece_type
        self.is_red = is_red
        
    def __str__(self):
        symbol = self.piece_type.upper() if self.is_red else self.piece_type.lower()
        return symbol

    def get_moves(self, pos, board):
        """Base method for getting valid moves"""
        raise NotImplementedError

    def calculate_destination(self, pos, movement, destination):
        """Calculate destination position based on movement and destination"""
        raise NotImplementedError

class Horse(Piece):
    def __init__(self, is_red):
        super().__init__('h', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        # Horse's L-shaped moves: 2 steps orthogonally then 1 step diagonally
        horse_moves = [
            (-2, -1), (-2, 1), (2, -1), (2, 1),
            (-1, -2), (-1, 2), (1, -2), (1, 2)
        ]
        
        for dr, dc in horse_moves:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_pos((new_row, new_col), board):
                # Check for blocking piece (马腿)
                block_row = row + (dr // 2)
                block_col = col + (dc // 2)
                if not board[block_row][block_col]:
                    target = board[new_row][new_col]
                    if not target or target.is_red != self.is_red:
                        moves.append(((row, col), (new_row, new_col)))
        return moves

    def _is_valid_pos(self, pos, board):
        row, col = pos
        return 0 <= row < 10 and 0 <= col < 9

    def calculate_destination(self, pos, movement, destination):
        """Calculate horse's destination based on movement"""
        row, col = pos
        dest = int(destination)
        # Convert target column based on perspective
        target_col = (9 - dest) if self.is_red else (dest - 1)
        
        if movement == "forward":
            # Move forward 2 (up for red, down for black)
            new_row = row + (2 if self.is_red else -2)
            return (new_row, target_col)
        elif movement == "backward":
            # Move backward 2 (down for red, up for black) 
            new_row = row + (-2 if self.is_red else 2)
            return (new_row, target_col)
        return None

class Rook(Piece):
    def __init__(self, is_red):
        super().__init__('r', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            while 0 <= new_row < 10 and 0 <= new_col < 9:
                target = board[new_row][new_col]
                if not target:
                    moves.append(((row, col), (new_row, new_col)))
                else:
                    if target.is_red != self.is_red:
                        moves.append(((row, col), (new_row, new_col)))
                    break
                new_row, new_col = new_row + dr, new_col + dc
        return moves

    def calculate_destination(self, pos, movement, destination):
        row, col = pos
        dest = int(destination)
        
        if movement == "horizontal":
            # Move horizontally to target column
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (row, new_col)
        elif movement in ["forward", "backward"]:
            # Move vertically by number of steps
            steps = dest
            if movement == "forward":
                new_row = row + (-steps if self.is_red else steps)
            else:  # backward
                new_row = row + (steps if self.is_red else -steps)
            return (new_row, col)
        return None

class Cannon(Piece):
    def __init__(self, is_red):
        super().__init__('c', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # First phase: moving without capture
            while 0 <= new_row < 10 and 0 <= new_col < 9:
                target = board[new_row][new_col]
                if not target:
                    moves.append(((row, col), (new_row, new_col)))
                else:
                    # Found first piece, look for capture
                    jump_row, jump_col = new_row + dr, new_col + dc
                    while 0 <= jump_row < 10 and 0 <= jump_col < 9:
                        jump_target = board[jump_row][jump_col]
                        if jump_target:
                            if jump_target.is_red != self.is_red:
                                moves.append(((row, col), (jump_row, jump_col)))
                            break
                        jump_row, jump_col = jump_row + dr, jump_col + dc
                    break
                new_row, new_col = new_row + dr, new_col + dc
        return moves

    def calculate_destination(self, pos, movement, destination):
        row, col = pos
        dest = int(destination)
        
        if movement == "horizontal":
            # Move horizontally to target column
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (row, new_col)
        elif movement in ["forward", "backward"]:
            # Move vertically by number of steps
            steps = dest
            if movement == "forward":
                new_row = row + (-steps if self.is_red else steps)
            else:  # backward
                new_row = row + (steps if self.is_red else -steps)
            return (new_row, col)
        return None

class Elephant(Piece):
    def __init__(self, is_red):
        super().__init__('e', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        moves_delta = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        for dr, dc in moves_delta:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 10 and 0 <= new_col < 9:
                # Check if elephant stays on its side
                if self.is_red and new_row <= 4 or not self.is_red and new_row >= 5:
                    # Check blocking piece
                    block_row = row + (dr // 2)
                    block_col = col + (dc // 2)
                    if not board[block_row][block_col]:
                        target = board[new_row][new_col]
                        if not target or target.is_red != self.is_red:
                            moves.append(((row, col), (new_row, new_col)))
        return moves

    def calculate_destination(self, pos, movement, destination):
        row, col = pos
        dest = int(destination)
        
        if movement == "forward":
            # Move diagonally forward
            new_row = row + (-2 if self.is_red else 2)
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (new_row, new_col)
        elif movement == "backward":
            # Move diagonally backward
            new_row = row + (2 if self.is_red else -2)
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (new_row, new_col)
        return None

class Advisor(Piece):
    def __init__(self, is_red):
        super().__init__('a', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        moves_delta = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in moves_delta:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 10 and 0 <= new_col < 9:
                # Check if move is within palace
                if self.is_red and 0 <= new_row <= 2 and 3 <= new_col <= 5 or \
                   not self.is_red and 7 <= new_row <= 9 and 3 <= new_col <= 5:
                    target = board[new_row][new_col]
                    if not target or target.is_red != self.is_red:
                        moves.append(((row, col), (new_row, new_col)))
        return moves

    def calculate_destination(self, pos, movement, destination):
        row, col = pos
        dest = int(destination)
        
        if movement == "forward":
            # Move diagonally forward
            new_row = row + (-1 if self.is_red else 1)
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (new_row, new_col)
        elif movement == "backward":
            # Move diagonally backward
            new_row = row + (1 if self.is_red else -1)
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (new_row, new_col)
        return None

class General(Piece):
    def __init__(self, is_red):
        super().__init__('k', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 10 and 0 <= new_col < 9:
                # Check if move is within palace
                if self.is_red and 0 <= new_row <= 2 and 3 <= new_col <= 5 or \
                   not self.is_red and 7 <= new_row <= 9 and 3 <= new_col <= 5:
                    target = board[new_row][new_col]
                    if not target or target.is_red != self.is_red:
                        moves.append(((row, col), (new_row, new_col)))
        return moves

    def calculate_destination(self, pos, movement, destination):
        row, col = pos
        dest = int(destination)
        
        if movement == "horizontal":
            # Move horizontally to target column
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (row, new_col)
        elif movement in ["forward", "backward"]:
            # Move vertically by 1 step
            if movement == "forward":
                new_row = row + (-1 if self.is_red else 1)
            else:  # backward
                new_row = row + (1 if self.is_red else -1)
            return (new_row, col)
        return None

class Pawn(Piece):
    def __init__(self, is_red):
        super().__init__('p', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        # Direction depends on color
        forward = 1 if self.is_red else -1
        
        # Forward move
        new_row = row + forward
        if 0 <= new_row < 10:
            target = board[new_row][col]
            if not target or target.is_red != self.is_red:
                moves.append(((row, col), (new_row, col)))
        
        # Horizontal moves if across river
        if (self.is_red and row > 4) or (not self.is_red and row < 5):
            for dc in [-1, 1]:
                new_col = col + dc
                if 0 <= new_col < 9:
                    target = board[row][new_col]
                    if not target or target.is_red != self.is_red:
                        moves.append(((row, col), (row, new_col)))
        return moves

    def calculate_destination(self, pos, movement, destination):
        row, col = pos
        dest = int(destination)
        
        if movement == "horizontal":
            # Move horizontally to target column (only after crossing river)
            new_col = (9 - dest) if self.is_red else (dest - 1)
            return (row, new_col)
        elif movement == "forward":
            # Move forward by 1 step
            new_row = row + (-1 if self.is_red else 1)
            return (new_row, col)
        return None

class XiangqiEnv:
    def __init__(self):
        self.board = [[None for _ in range(9)] for _ in range(10)]
        self.current_player = True  # True for red, False for black
        self.reset()
    
    def reset(self):
        # Initialize the board with pieces
        # Black pieces (top)
        self.board[0] = [
            Rook(False), Horse(False), Elephant(False),
            Advisor(False), General(False), Advisor(False),
            Elephant(False), Horse(False), Rook(False)
        ]
        self.board[2][1] = Cannon(False)
        self.board[2][7] = Cannon(False)
        for i in [0, 2, 4, 6, 8]:
            self.board[3][i] = Pawn(False)

        # Red pieces (bottom)
        self.board[9] = [
            Rook(True), Horse(True), Elephant(True),
            Advisor(True), General(True), Advisor(True),
            Elephant(True), Horse(True), Rook(True)
        ]
        self.board[7][1] = Cannon(True)
        self.board[7][7] = Cannon(True)
        for i in [0, 2, 4, 6, 8]:
            self.board[6][i] = Pawn(True)

        self.current_player = True  # Red moves first
        return self.board
    
    def _get_state(self):
        """Convert board to neural network input"""
        state = np.zeros((14, 10, 9), dtype=np.float32)
        
        piece_channels = {
            'r': 0, 'h': 1, 'e': 2, 'a': 3, 'k': 4, 'c': 5, 'p': 6
        }
        
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece:
                    channel = piece_channels[piece.piece_type]
                    if not piece.is_red:
                        channel += 7
                    state[channel][i][j] = 1
        
        # Current player channel
        state[13] = np.full((10, 9), self.current_player, dtype=np.float32)
        
        return state
    
    def _is_valid_pos(self, pos):
        """Check if position is within board"""
        row, col = pos
        return 0 <= row < 10 and 0 <= col < 9
    
    def _is_in_palace(self, pos, is_red):
        """Check if position is within palace"""
        row, col = pos
        if is_red:
            return 0 <= row <= 2 and 3 <= col <= 5
        else:
            return 7 <= row <= 9 and 3 <= col <= 5
    
    def _get_piece_moves(self, pos):
        """Get valid moves for piece at position"""
        row, col = pos
        piece = self.board[row][col]
        if not piece or piece.is_red != self.current_player:
            return []
        
        moves = []
        
        if piece.piece_type == 'k':  # King/General
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if self._is_valid_pos((new_row, new_col)) and \
                   self._is_in_palace((new_row, new_col), piece.is_red):
                    target = self.board[new_row][new_col]
                    if not target or target.is_red != piece.is_red:
                        moves.append(((row, col), (new_row, new_col)))
        
        elif piece.piece_type == 'r':  # Rook
            moves = Rook(piece.is_red).get_moves((row, col), self.board)
        
        elif piece.piece_type == 'h':  # Horse
            moves = Horse(piece.is_red).get_moves((row, col), self.board)
        
        elif piece.piece_type == 'c':  # Cannon
            moves = Cannon(piece.is_red).get_moves((row, col), self.board)
        
        elif piece.piece_type == 'e':  # Elephant
            moves = Elephant(piece.is_red).get_moves((row, col), self.board)

        elif piece.piece_type == 'a':  # Advisor
            moves = Advisor(piece.is_red).get_moves((row, col), self.board)

        elif piece.piece_type == 'p':  # Pawn
            moves = Pawn(piece.is_red).get_moves((row, col), self.board)
        
        return moves
    
    def get_valid_moves(self):
        """Get all valid moves for current player"""
        moves = []
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.is_red == self.current_player:
                    moves.extend(self._get_piece_moves((i, j)))
        return moves
    
    def step(self, action):
        """Execute move and return new state"""
        from_pos, to_pos = action
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Check if move is valid
        if action not in self.get_valid_moves():
            return self._get_state(), -1, True
        
        # Move the piece
        piece = self.board[from_row][from_col]
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        # Switch players
        self.current_player = not self.current_player
        
        # Simple reward: 1 for winning, -1 for losing, 0 otherwise
        reward = 0
        done = False
        
        return self._get_state(), reward, done

    def calculate_move(self, move):
        """Convert Chinese notation move to board positions"""
        piece = move['piece']
        movement = move['movement']
        end = move['end']
        # Check if it's a red piece using the traditional characters
        is_red = piece in "车马象士将炮兵"  # Red uses simplified Chinese
        is_black = piece in "車馬相仕帥砲卒"  # Black uses traditional Chinese
        
        if not (is_red or is_black):
            print(f"Warning: Unknown piece character: {piece}")
            return None, None

        # Get start position
        if 'position_marker' in move:
            from_pos = self._find_piece_by_marker(piece, move['position_marker'])
        else:
            from_pos = self._find_piece_by_column(piece, move['start'])

        # Get end position
        if not from_pos:
            return None, None

        to_pos = self._calculate_destination(from_pos, piece, movement, end)
        return from_pos, to_pos

    def _find_piece_by_marker(self, piece, marker):
        """Find piece position using front/back marker"""
        pieces_pos = []
        piece_type = self._piece_to_type(piece)
        is_red = piece in "车马象士将炮兵"

        for i in range(10):
            for j in range(9):
                current_piece = self.board[i][j]
                if current_piece and current_piece.piece_type == piece_type:
                    if (current_piece.is_red and is_red) or \
                       (not current_piece.is_red and not is_red):
                        pieces_pos.append((i, j))

        if not pieces_pos:
            return None

        # Sort based on the marker
        if marker == "front":
            pieces_pos.sort(key=lambda x: x[0], reverse=is_red)
        elif marker == "back":
            pieces_pos.sort(key=lambda x: x[0], reverse=not is_red)

        return pieces_pos[0]

    def _find_piece_by_column(self, piece, col):
        """Find piece position using column number"""
        piece_type = self._piece_to_type(piece)
        is_red = piece in "车马象士将炮兵"  # Red uses simplified Chinese
        
        # Convert column number based on perspective
        col = (9 - int(col)) if is_red else (int(col) - 1)
        
        # Search in appropriate direction
        if is_red:
            # Red pieces are at the bottom, search from bottom up
            rows = range(9, -1, -1)
        else:
            # Black pieces are at the top, search from top down
            rows = range(0, 10)
        
        for row in rows:
            current_piece = self.board[row][col]
            if current_piece and current_piece.piece_type == piece_type:
                if (current_piece.is_red and is_red) or \
                   (not current_piece.is_red and not is_red):
                    return (row, col)
        return None

    def _calculate_destination(self, from_pos, piece, movement, destination):
        """Calculate destination position based on movement type"""
        row, col = from_pos
        current_piece = self.board[row][col]
        return current_piece.calculate_destination(from_pos, movement, destination)

    def _piece_to_type(self, piece):
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