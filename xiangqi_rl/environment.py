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
    
    def calculate_destination(self, pos, movement, destination):
        """Calculate horse's destination based on movement"""
        row, col = pos
        dest = int(destination)
        # Convert target column based on perspective
        target_col = (9 - dest) if self.is_red else (dest - 1)
        # Calculate row change based on movement direction
        if movement == "forward":
            # Move forward (up for red, down for black)
            row_change = -1 if self.is_red else 1
        elif movement == "backward":
            # Move backward (down for red, up for black)
            row_change = 1 if self.is_red else -1
        else:
            return None
            
        # Double the row change if col_change is 1 (马走日)
        if abs(target_col - col) == 1:
            row_change *= 2
        # Calculate possible destinations based on target column
        col_change = target_col - col
        
        # Check if the move is valid (马走日)
        if abs(row_change) == 2 and abs(col_change) == 1:
            return (row + row_change, target_col)
        elif abs(row_change) == 1 and abs(col_change) == 2:
            return (row + row_change, target_col)
            
        return None

    def get_moves(self, pos, board):
        """Get all valid moves for the horse"""
        row, col = pos
        moves = []
        # Horse's L-shaped moves: 2 steps orthogonally then 1 step diagonally
        horse_moves = [
            (-2, -1), (-2, 1),  # Forward 2, left/right 1
            (2, -1), (2, 1),    # Backward 2, left/right 1
            (-1, -2), (-1, 2),  # Forward 1, left/right 2
            (1, -2), (1, 2)     # Backward 1, left/right 2
        ]
        
        for dr, dc in horse_moves:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_pos((new_row, new_col), board):
                # Check for blocking piece (马腿)
                if abs(dr) == 2:  # Moving vertically first
                    block_row = row + (dr // 2)
                    block_col = col
                else:  # Moving horizontally first
                    block_row = row
                    block_col = col + (dc // 2)
                
                if not board[block_row][block_col]:  # No blocking piece
                    target = board[new_row][new_col]
                    if not target or target.is_red != self.is_red:
                        moves.append(((row, col), (new_row, new_col)))
        return moves

    def _is_valid_pos(self, pos, board):
        row, col = pos
        return 0 <= row < 10 and 0 <= col < 9

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
                if self.is_red and new_row >= 5 or not self.is_red and new_row <= 4:
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
    
    def calculate_destination(self, pos, movement, destination):
        """Calculate advisor's destination based on movement"""
        row, col = pos
        dest = int(destination)
        
        # Convert target column based on perspective
        target_col = (9 - dest) if self.is_red else (dest - 1)
        
        # Calculate row change based on movement direction
        if movement == "forward":
            # Move forward diagonally (up for red, down for black)
            row_change = -1 if self.is_red else 1
        elif movement == "backward":
            # Move backward diagonally (down for red, up for black)
            row_change = 1 if self.is_red else -1
        else:
            return None
        
        # Calculate new position
        new_row = row + row_change
        
        # Advisor must move diagonally one step
        # The target column must be adjacent to current column
        if abs(target_col - col) == 1:
            new_pos = (new_row, target_col)
            # Verify the move is within palace
            if self._is_in_palace(new_pos):
                return new_pos
        
        return None

    def get_moves(self, pos, board):
        """Get all valid moves for the advisor"""
        row, col = pos
        moves = []
        # Advisor moves diagonally within palace
        moves_delta = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in moves_delta:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_pos((new_row, new_col), board) and \
               self._is_in_palace((new_row, new_col)):
                target = board[new_row][new_col]
                if not target or target.is_red != self.is_red:
                    moves.append(((row, col), (new_row, new_col)))
        return moves

    def _is_valid_pos(self, pos, board):
        """Check if position is within board"""
        row, col = pos
        return 0 <= row < 10 and 0 <= col < 9

    def _is_in_palace(self, pos):
        """Check if position is within the palace"""
        row, col = pos
        if self.is_red:
            return 7 <= row <= 9 and 3 <= col <= 5
        else:
            return 0 <= row <= 2 and 3 <= col <= 5

class General(Piece):
    def __init__(self, is_red):
        super().__init__('k', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_pos((new_row, new_col), board) and \
               self._is_in_palace((new_row, new_col)):
                target = board[new_row][new_col]
                if not target or target.is_red != self.is_red:
                    moves.append(((row, col), (new_row, new_col)))
        return moves

    def calculate_destination(self, pos, movement, destination):
        """Calculate general's destination based on movement"""
        row, col = pos
        dest = int(destination)
        
        # Convert target column based on perspective
        target_col = (9 - dest) if self.is_red else (dest - 1)
        
        # Calculate new position based on movement type
        if movement == "horizontal":
            new_pos = (row, target_col)
        elif movement == "forward":
            new_row = row + (-1 if self.is_red else 1)
            new_pos = (new_row, col)
        elif movement == "backward":
            new_row = row + (1 if self.is_red else -1)
            new_pos = (new_row, col)
        else:
            return None
            
        # Verify the move is within palace
        if self._is_in_palace(new_pos):
            return new_pos
        return None

    def _is_valid_pos(self, pos, board):
        """Check if position is within board"""
        row, col = pos
        return 0 <= row < 10 and 0 <= col < 9

    def _is_in_palace(self, pos):
        """Check if position is within the palace"""
        row, col = pos
        if self.is_red:
            return 7 <= row <= 9 and 3 <= col <= 5
        else:
            return 0 <= row <= 2 and 3 <= col <= 5

class Pawn(Piece):
    def __init__(self, is_red):
        super().__init__('p', is_red)
    
    def get_moves(self, pos, board):
        row, col = pos
        moves = []
        # Direction depends on color
        forward = -1 if self.is_red else 1
        
        # Forward move
        new_row = row + forward
        if 0 <= new_row < 10:
            target = board[new_row][col]
            if not target or target.is_red != self.is_red:
                moves.append(((row, col), (new_row, col)))
        
        # Horizontal moves if across river
        if (self.is_red and row < 5) or (not self.is_red and row > 4):
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
        self.action_space_size = 90 * 90  # All possible moves (from any square to any square)
        self.reset()
    
    def reset(self):
        """重置棋盘和所有状态值"""
        # 重置棋盘
        self.board = [[None for _ in range(9)] for _ in range(10)]
        
        # 初始化黑方棋子（上方）
        self.board[0] = [
            Rook(False), Horse(False), Elephant(False),
            Advisor(False), General(False), Advisor(False),
            Elephant(False), Horse(False), Rook(False)
        ]
        self.board[2][1] = Cannon(False)
        self.board[2][7] = Cannon(False)
        for i in [0, 2, 4, 6, 8]:
            self.board[3][i] = Pawn(False)

        # 初始化红方棋子（下方）
        self.board[9] = [
            Rook(True), Horse(True), Elephant(True),
            Advisor(True), General(True), Advisor(True),
            Elephant(True), Horse(True), Rook(True)
        ]
        self.board[7][1] = Cannon(True)
        self.board[7][7] = Cannon(True)
        for i in [0, 2, 4, 6, 8]:
            self.board[6][i] = Pawn(True)

        # 重置游戏状态
        self.current_player = True  # 红方先手
        self.move_count = 0  # 重置移动计数
        self.last_move = None  # 记录最后一步移动
        self.history = []  # 记录游戏历史
        self.captured_pieces = {
            True: [],   # 红方被吃的子
            False: []   # 黑方被吃的子
        }
        
        # 重置游戏结果相关状态
        self.is_game_over = False
        self.winner = None
        
        return self._get_state()
    
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
        
        # Filter out moves that would result in flying general
        valid_moves = []
        for move in moves:
            # Simulate the move
            from_pos, to_pos = move
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            
            # Store original pieces
            moving_piece = self.board[from_row][from_col]
            captured_piece = self.board[to_row][to_col]
            
            # Make move
            self.board[to_row][to_col] = moving_piece
            self.board[from_row][from_col] = None
            
            # Check if kings face each other
            if not self._kings_face_each_other():
                valid_moves.append(move)
            
            # Restore board
            self.board[from_row][from_col] = moving_piece
            self.board[to_row][to_col] = captured_piece
        
        return valid_moves
    
    def _kings_face_each_other(self):
        """Check if the two kings face each other directly"""
        # Find positions of both kings
        red_king_pos = None
        black_king_pos = None
        
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.piece_type == 'k':
                    if piece.is_red:
                        red_king_pos = (i, j)
                    else:
                        black_king_pos = (i, j)
        
        if not (red_king_pos and black_king_pos):
            return False
        
        # Check if kings are in the same column
        red_row, red_col = red_king_pos
        black_row, black_col = black_king_pos
        
        if red_col != black_col:
            return False
        
        # Check if there are any pieces between them
        min_row = min(red_row, black_row)
        max_row = max(red_row, black_row)
        
        for row in range(min_row + 1, max_row):
            if self.board[row][red_col]:
                return False
        
        return True
    
    def _is_in_check(self):
        """Check if current player's king is in check"""
        return self.is_king_in_check(self.current_player)

    def _is_checkmate(self):
        """Check if current player is in checkmate"""
        # First check if king is in check
        if not self._is_in_check():
            return False
        
        # Try all possible moves to see if check can be escaped
        original_player = self.current_player
        for move in self.get_valid_moves():
            # Save board state
            from_pos, to_pos = move
            from_piece = self.board[from_pos[0]][from_pos[1]]
            to_piece = self.board[to_pos[0]][to_pos[1]]
            
            # Try move
            self.board[to_pos[0]][to_pos[1]] = from_piece
            self.board[from_pos[0]][from_pos[1]] = None
            
            # Check if still in check
            still_in_check = self._is_in_check()
            
            # Restore board state
            self.board[from_pos[0]][from_pos[1]] = from_piece
            self.board[to_pos[0]][to_pos[1]] = to_piece
            
            if not still_in_check:
                return False
        
        return True

    def _is_stalemate(self):
        """Check if current player is in stalemate"""
        return not self._is_in_check() and not self.get_valid_moves()

    def step(self, action):
        """Execute one step in the environment"""
        (from_row, from_col), (to_row, to_col) = action
        piece = self.board[from_row][from_col]
        target = self.board[to_row][to_col]
        
        # Move piece
        self.board[from_row][from_col] = None
        self.board[to_row][to_col] = piece
        
        # Record captured piece
        if target:
            self.captured_pieces[target.is_red].append(target)
            # Check if captured piece was a general/king
            if target.piece_type == 'k':
                self.is_game_over = True
                self.winner = piece.is_red  # The player who captured the king wins
        
        # Record move
        self.last_move = action
        self.history.append(action)
        
        # Switch player
        self.current_player = not self.current_player
        self.move_count += 1
        
        # Check if current player has any valid moves
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            self.is_game_over = True
            self.winner = not self.current_player  # Current player has no moves, so opponent wins
        
        # Check for general face-to-face
        if self._generals_face_to_face():
            self.is_game_over = True
            self.winner = not self.current_player  # Previous player wins
        
        return self._get_state()

    def _generals_face_to_face(self):
        """Check if the two generals are facing each other directly"""
        # Find positions of both generals
        red_general_pos = None
        black_general_pos = None
        
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.piece_type == 'k':
                    if piece.is_red:
                        red_general_pos = (i, j)
                    else:
                        black_general_pos = (i, j)
        
        if not (red_general_pos and black_general_pos):
            return False
        
        # Check if they're in the same column
        if red_general_pos[1] != black_general_pos[1]:
            return False
        
        # Check if there are any pieces between them
        start = min(red_general_pos[0], black_general_pos[0])
        end = max(red_general_pos[0], black_general_pos[0])
        
        for i in range(start + 1, end):
            if self.board[i][red_general_pos[1]]:
                return False
        
        return True

    def is_king_in_check(self, is_red):
        """Check if the king of the given color is in check"""
        # Find king position
        king_pos = None
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.piece_type == 'k' and piece.is_red == is_red:
                    king_pos = (i, j)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return True  # King is captured
        
        # Check if any opponent piece can capture the king
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.is_red != is_red:
                    moves = piece.get_moves((i, j), self.board)
                    if king_pos in [move[1] for move in moves]:
                        return True
        
        return False

    def calculate_move(self, move, is_red):
        """Convert Chinese notation move to board positions
        Args:
            move: The move in Chinese notation
            is_red: True if it's red's turn, False if it's black's turn
        """
        piece = move['piece']
        movement = move['movement']
        end = move['end']
        
        # Get start position
        if 'position_marker' in move:
            from_pos = self._find_piece_by_marker(piece, move['position_marker'], is_red)
        else:
            from_pos = self._find_piece_by_column(piece, move['start'], is_red)

        # Get end position
        if not from_pos:
            return None, None

        to_pos = self._calculate_destination(from_pos, piece, movement, end)
        return from_pos, to_pos

    def _find_piece_by_marker(self, piece, marker, is_red):
        """Find piece position using front/back marker"""
        pieces_pos = []
        piece_type = self._piece_to_type(piece)

        for i in range(10):
            for j in range(9):
                current_piece = self.board[i][j]
                if current_piece and current_piece.piece_type == piece_type:
                    if current_piece.is_red == is_red:
                        pieces_pos.append((i, j))

        if not pieces_pos:
            return None

        # Sort based on the marker
        if marker == "front":
            pieces_pos.sort(key=lambda x: x[0], reverse=not is_red)
        elif marker == "back":
            pieces_pos.sort(key=lambda x: x[0], reverse=is_red)
        return pieces_pos[0]

    def _find_piece_by_column(self, piece, col, is_red):
        """Find piece position using column number"""
        piece_type = self._piece_to_type(piece)
        
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
                if current_piece.is_red == is_red:
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

    def _get_threatened_pieces(self, pos):
        """获取在给定位置可以威胁到的对方棋子"""
        row, col = pos
        piece = self.board[row][col]
        if not piece:
            return []
        
        threatened = []
        valid_moves = piece.get_moves(pos, self.board)
        for _, (to_row, to_col) in valid_moves:
            target = self.board[to_row][to_col]
            if target and target.is_red != piece.is_red:
                threatened.append(target)
        return threatened

    def _get_protected_pieces(self, pos):
        """获取在给定位置可以保护到的己方棋子"""
        row, col = pos
        piece = self.board[row][col]
        if not piece:
            return []
        
        protected = []
        valid_moves = piece.get_moves(pos, self.board)
        for _, (to_row, to_col) in valid_moves:
            target = self.board[to_row][to_col]
            if target and target.is_red == piece.is_red:
                protected.append(target)
        return protected

    def get_canonical_state(self):
        """Get canonical form of the state (from current player's perspective)"""
        state = self._get_state()
        
        if not self.current_player:  # If it's black's turn
            # Flip the board vertically
            state = np.flip(state, axis=1)
            # Swap red and black piece channels
            for i in range(7):
                state[i], state[i+7] = np.copy(state[i+7]), np.copy(state[i])
            # Flip the current player channel
            state[13] = 1 - state[13]
        
        # Convert to immutable type for dictionary key
        return tuple(map(tuple, state.reshape(-1, 9))) 

    def clone(self):
        """Create a deep copy of the environment"""
        new_env = XiangqiEnv()
        
        # Copy board state
        new_env.board = [[None for _ in range(9)] for _ in range(10)]
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece:
                    # Create new piece instance with same properties
                    new_piece = type(piece)(piece.is_red)
                    new_env.board[i][j] = new_piece
        
        # Copy other state variables
        new_env.current_player = self.current_player
        new_env.move_count = self.move_count
        new_env.is_game_over = self.is_game_over
        new_env.winner = self.winner
        
        # Don't need to copy history or captured pieces for MCTS simulation
        
        return new_env 

    def get_state(self):
        """Public method to get the current state"""
        return self._get_state() 

    def get_reward(self):
        """Get the reward for the current state"""
        if not self.is_game_over:
            return 0
        
        if self.winner is None:  # Draw
            return 0
        # Return 1 for win, -1 for loss from current player's perspective
        return 1 if self.winner == self.current_player else -1 