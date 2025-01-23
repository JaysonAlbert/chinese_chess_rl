import pygame
import numpy as np

class XiangqiVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.square_size = 60
        self.width = 9 * self.square_size
        self.height = 10 * self.square_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Chinese Chess')
        
        # Colors
        self.BACKGROUND = (255, 223, 162)
        self.LINES = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        
        # Piece names
        self.piece_names = {
            1: 'R車', 2: 'R馬', 3: 'R象', 4: 'R士', 5: 'R帥',
            6: 'R炮', 7: 'R兵',
            -1: 'B車', -2: 'B馬', -3: 'B象', -4: 'B士', -5: 'B將',
            -6: 'B炮', -7: 'B卒'
        }
        
        # Initialize font
        self.font = pygame.font.SysFont('simsun', 30)
    
    def draw_board(self):
        self.screen.fill(self.BACKGROUND)
        
        # Draw lines
        for i in range(10):
            pygame.draw.line(self.screen, self.LINES,
                           (0, i * self.square_size),
                           (self.width, i * self.square_size))
        for i in range(9):
            pygame.draw.line(self.screen, self.LINES,
                           (i * self.square_size, 0),
                           (i * self.square_size, self.height))
        
        # Draw pieces
        for i in range(10):
            for j in range(9):
                piece = self.env.board[i, j]
                if piece != 0:
                    color = self.RED if piece > 0 else self.BLACK
                    text = self.font.render(self.piece_names[piece][1], True, color)
                    x = j * self.square_size + self.square_size // 2 - text.get_width() // 2
                    y = i * self.square_size + self.square_size // 2 - text.get_height() // 2
                    self.screen.blit(text, (x, y))
        
        pygame.display.flip() 