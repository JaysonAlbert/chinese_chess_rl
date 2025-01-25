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
        """检查当前玩家是否被将军"""
        # 找到当前玩家的将/帅
        king_pos = None
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.piece_type == 'k' and piece.is_red == self.current_player:
                    king_pos = (i, j)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # 如果找不到将/帅，返回False
        
        # 检查对手的所有棋子是否可以攻击到将/帅
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if piece and piece.is_red != self.current_player:  # 对手的棋子
                    valid_moves = piece.get_moves((i, j), self.board)
                    for _, to_pos in valid_moves:
                        if to_pos == king_pos:  # 可以攻击到将/帅
                            return True
        
        return False

    def _is_checkmate(self):
        """检查是否将死"""
        # 首先检查是否被将军
        if not self._is_in_check():
            return False
        
        # 获取所有可能的移动
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return True  # 无法移动，被将死
        
        # 尝试每一个可能的移动，看是否能解除将军
        for move in valid_moves:
            # 保存当前状态
            from_pos, to_pos = move
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            moving_piece = self.board[from_row][from_col]
            captured_piece = self.board[to_row][to_col]
            
            # 尝试移动
            self.board[to_row][to_col] = moving_piece
            self.board[from_row][from_col] = None
            
            # 检查是否仍然被将军
            still_in_check = self._is_in_check()
            
            # 恢复状态
            self.board[from_row][from_col] = moving_piece
            self.board[to_row][to_col] = captured_piece
            
            if not still_in_check:
                return False  # 找到一个可以解除将军的移动
        
        return True  # 所有移动都无法解除将军，确实被将死

    def _is_stalemate(self):
        """检查是否和棋"""
        # 如果被将军但没被将死，不是和棋
        if self._is_in_check():
            return False
        
        # 如果没有合法移动，是和棋
        valid_moves = self.get_valid_moves()
        return len(valid_moves) == 0

    def step(self, action):
        """执行移动并返回新状态"""
        from_pos, to_pos = action
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 获取棋子并检查是否属于当前玩家
        piece = self.board[from_row][from_col]
        if not piece or piece.is_red != self.current_player:
            return self._get_state(), -1, True
            
        # Check if move is valid for this piece
        valid_moves = piece.get_moves(from_pos, self.board)
        if action not in valid_moves:
            return self._get_state(), -1, True
        
        # Check if capturing opponent's general/king
        captured_piece = self.board[to_row][to_col]
        is_winning_move = captured_piece and captured_piece.piece_type == 'k'
        
        # Move the piece
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        # Switch players
        self.current_player = not self.current_player
        
        # Check if next player has any valid moves
        next_valid_moves = self.get_valid_moves()
        is_checkmate = len(next_valid_moves) == 0
        
        # Game is over if king is captured or checkmate
        done = is_winning_move or is_checkmate
        
        # 调整奖励尺度
        piece_values = {
            'p': 1,   # 兵/卒
            'c': 4,   # 炮
            'h': 4,   # 马
            'r': 9,   # 车
            'a': 2,   # 士
            'e': 2,   # 象
            'k': 100  # 将/帅
        }
        
        reward = 0
        # 1. 吃子奖励 - 增加基础奖励
        if captured_piece:
            reward += piece_values.get(captured_piece.piece_type, 0) * 0.3  # 从0.1增加到0.3
        
        # 2. 位置奖励 - 增加中心控制奖励
        central_positions = [(4, 1), (4, 2), (4, 7), (4, 8)]
        if (to_row, to_col) in central_positions:
            reward += 0.2  # 从0.05增加到0.2
        
        # 3. 威胁奖励 - 增加威胁奖励
        threatened_pieces = self._get_threatened_pieces((to_row, to_col))
        for piece in threatened_pieces:
            reward += piece_values.get(piece.piece_type, 0) * 0.15  # 从0.05增加到0.15
        
        # 4. 保护奖励 - 增加保护奖励
        protected_pieces = self._get_protected_pieces((to_row, to_col))
        for piece in protected_pieces:
            reward += piece_values.get(piece.piece_type, 0) * 0.1  # 从0.02增加到0.1
        
        # 5. 游戏结束奖励 - 调整胜负奖励
        if is_checkmate and is_winning_move:
            reward = 5.0  # 从1.0增加到5.0
        elif is_checkmate:
            reward = -3.0  # 从-1.0改为-3.0，减少惩罚
        elif self._is_in_check():
            reward += 0.5  # 从0.1增加到0.5
        elif self._is_stalemate():
            reward = -0.5  # 从0改为-0.5，轻微惩罚和棋
        
        # 6. 惩罚过长的游戏 - 减少惩罚力度
        if hasattr(self, 'move_count'):
            self.move_count += 1
        else:
            self.move_count = 1
        
        if self.move_count > 200:
            reward -= 0.0005 * (self.move_count - 200)  # 从0.001减少到0.0005
        
        # 7. 添加移动有效性奖励
        if action in valid_moves:
            reward += 0.1  # 奖励有效移动
        
        # 8. 添加进攻性奖励
        if to_row < 5 and self.current_player:  # 红方过河
            reward += 0.2
        elif to_row > 4 and not self.current_player:  # 黑方过河
            reward += 0.2
        
        # 记录移动历史
        self.last_move = action
        self.history.append({
            'action': action,
            'piece': piece.piece_type,
            'is_red': piece.is_red,
            'captured': captured_piece.piece_type if captured_piece else None
        })
        
        # 如果有子被吃，记录到被吃子列表
        if captured_piece:
            self.captured_pieces[captured_piece.is_red].append(captured_piece)
        
        # 更新游戏结果状态
        if is_checkmate and is_winning_move:
            self.is_game_over = True
            self.winner = self.current_player
        elif is_checkmate:
            self.is_game_over = True
            self.winner = not self.current_player
        elif self._is_stalemate():
            self.is_game_over = True
            self.winner = None
        
        return self._get_state(), reward, done

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