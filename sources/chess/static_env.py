from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import List, Tuple, Optional, Set, Dict, Union
from logging import getLogger

logger = getLogger(__name__)

class PieceType(Enum):
    ROOK = "R"
    KNIGHT = "N" 
    BISHOP = "B"
    ADVISOR = "A"
    KING = "K"
    CANNON = "C"
    PAWN = "P"

class Color(Enum):
    RED = auto()
    BLACK = auto()

class GameResult(Enum):
    RED_WIN = 1
    BLACK_WIN = -1
    ONGOING = 0

@dataclass(frozen=True)
class Position:
    x: int
    y: int
    
    def __post_init__(self):
        if not (0 <= self.x < 9 and 0 <= self.y < 10):
            raise ValueError(f"Invalid position: ({self.x}, {self.y})")
    
    def flip(self) -> Position:
        return Position(8 - self.x, 9 - self.y)
    
    def in_palace(self, color: Color) -> bool:
        if not (3 <= self.x <= 5):
            return False
        if color == Color.RED:
            return self.y <= 2
        else:
            return self.y >= 7

@dataclass(frozen=True)
class Move:
    from_pos: Position
    to_pos: Position
    
    def __str__(self) -> str:
        return f"{self.from_pos.x}{self.from_pos.y}{self.to_pos.x}{self.to_pos.y}"
    
    def flip(self) -> Move:
        return Move(self.from_pos.flip(), self.to_pos.flip())

@dataclass(frozen=True)
class Piece:
    type: PieceType
    color: Color
    
    @property
    def value(self) -> int:
        values = {
            PieceType.ROOK: 14,
            PieceType.KING: 7,
            PieceType.BISHOP: 3,
            PieceType.KNIGHT: 2,
            PieceType.ADVISOR: 1,
            PieceType.CANNON: 5,
            PieceType.PAWN: 1
        }
        return values[self.type]
    
    @property
    def symbol(self) -> str:
        symbol = self.type.value
        return symbol if self.color == Color.RED else symbol.lower()
    
    @classmethod
    def from_symbol(cls, symbol: str) -> Optional[Piece]:
        if not symbol.isalpha():
            return None
        
        type_map = {
            'R': PieceType.ROOK, 'r': PieceType.ROOK,
            'N': PieceType.KNIGHT, 'n': PieceType.KNIGHT,
            'B': PieceType.BISHOP, 'b': PieceType.BISHOP,
            'A': PieceType.ADVISOR, 'a': PieceType.ADVISOR,
            'K': PieceType.KING, 'k': PieceType.KING,
            'C': PieceType.CANNON, 'c': PieceType.CANNON,
            'P': PieceType.PAWN, 'p': PieceType.PAWN
        }
        
        piece_type = type_map.get(symbol)
        if not piece_type:
            return None
            
        color = Color.RED if symbol.isupper() else Color.BLACK
        return cls(piece_type, color)

class MoveStrategy(ABC):
    @abstractmethod
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        pass

class PawnMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        piece = board.get_piece(pos)
        if not piece or piece.type != PieceType.PAWN:
            return moves
            
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_x = pos.x + dx
            new_y = pos.y + dy
            
            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                continue
                
            if piece.color == Color.BLACK and pos.y >= 5 and dx != 0:
                continue
                
            new_pos = Position(new_x, new_y)
            if board.can_move_to(new_pos, piece.color):
                moves.append(Move(pos, new_pos))
                
        return moves

class RookMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            for i in range(1, 10):
                new_x = pos.x + dx * i
                new_y = pos.y + dy * i
                
                if not (0 <= new_x < 9 and 0 <= new_y < 10):
                    break
                
                new_pos = Position(new_x, new_y)
                target_piece = board.get_piece(new_pos)
                if target_piece:
                    if target_piece.color != board.get_piece(pos).color:
                        moves.append(Move(pos, new_pos))
                    break
                else:
                    moves.append(Move(pos, new_pos))
                    
        return moves

class CannonMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            platform_found = False
            for i in range(1, 10):
                new_x = pos.x + dx * i
                new_y = pos.y + dy * i
                
                if not (0 <= new_x < 9 and 0 <= new_y < 10):
                    break
                
                new_pos = Position(new_x, new_y)
                target_piece = board.get_piece(new_pos)
                if not platform_found:
                    if target_piece:
                        platform_found = True
                    else:
                        moves.append(Move(pos, new_pos))
                else:
                    if target_piece:
                        if target_piece.color != board.get_piece(pos).color:
                            moves.append(Move(pos, new_pos))
                        break
                        
        return moves

class KnightMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        knight_moves = [
            (2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
            (1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1), (-1, -2, 0, -1)
        ]
        
        for dx, dy, block_x, block_y in knight_moves:
            block_pos_x = pos.x + block_x
            block_pos_y = pos.y + block_y
            
            if not (0 <= block_pos_x < 9 and 0 <= block_pos_y < 10):
                continue
                
            block_pos = Position(block_pos_x, block_pos_y)
            if board.get_piece(block_pos):
                continue
                
            new_x = pos.x + dx
            new_y = pos.y + dy
            
            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                continue
                
            new_pos = Position(new_x, new_y)
            if board.can_move_to(new_pos, board.get_piece(pos).color):
                moves.append(Move(pos, new_pos))
                
        return moves

class BishopMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        piece = board.get_piece(pos)
        bishop_moves = [(2, 2, 1, 1), (2, -2, 1, -1), (-2, 2, -1, 1), (-2, -2, -1, -1)]
        
        for dx, dy, block_x, block_y in bishop_moves:
            block_pos_x = pos.x + block_x
            block_pos_y = pos.y + block_y
            
            if not (0 <= block_pos_x < 9 and 0 <= block_pos_y < 10):
                continue
                
            block_pos = Position(block_pos_x, block_pos_y)
            if board.get_piece(block_pos):
                continue
                
            new_x = pos.x + dx
            new_y = pos.y + dy
            
            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                continue
                
            new_pos = Position(new_x, new_y)
            
            if piece.color == Color.BLACK and new_y <= 4:
                continue
                
            if board.can_move_to(new_pos, piece.color):
                moves.append(Move(pos, new_pos))
                
        return moves

class KingMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        piece = board.get_piece(pos)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_x = pos.x + dx
            new_y = pos.y + dy
            
            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                continue
                
            new_pos = Position(new_x, new_y)
            if new_pos.in_palace(piece.color) and board.can_move_to(new_pos, piece.color):
                moves.append(Move(pos, new_pos))
        
        if piece.color == Color.BLACK:
            for y in range(pos.y + 1, 10):
                target_pos = Position(pos.x, y)
                target_piece = board.get_piece(target_pos)
                if target_piece:
                    if target_piece.type == PieceType.KING and target_piece.color == Color.RED:
                        moves.append(Move(pos, target_pos))
                    break
                    
        return moves

class AdvisorMoveStrategy(MoveStrategy):
    def get_moves(self, board: ChessBoard, pos: Position) -> List[Move]:
        moves = []
        piece = board.get_piece(pos)
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            new_x = pos.x + dx
            new_y = pos.y + dy
            
            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                continue
                
            new_pos = Position(new_x, new_y)
            if (new_pos.in_palace(piece.color) and
                board.can_move_to(new_pos, piece.color)):
                moves.append(Move(pos, new_pos))
                
        return moves

class ChessBoard:
    INIT_STATE = 'rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR'
    
    def __init__(self, state: str = None):
        self._board: np.ndarray = np.full((10, 9), None, dtype=object)
        self._state_cache: Dict[str, any] = {}
        self._move_strategies: Dict[PieceType, MoveStrategy] = {
            PieceType.PAWN: PawnMoveStrategy(),
            PieceType.ROOK: RookMoveStrategy(),
            PieceType.CANNON: CannonMoveStrategy(),
            PieceType.KNIGHT: KnightMoveStrategy(),
            PieceType.BISHOP: BishopMoveStrategy(),
            PieceType.KING: KingMoveStrategy(),
            PieceType.ADVISOR: AdvisorMoveStrategy()
        }
        
        if state:
            self._load_from_state(state)
        else:
            self._load_from_state(self.INIT_STATE)
    
    def _load_from_state(self, state: str) -> None:
        self._board.fill(None)
        rows = state.split('/')
        
        for i, row in enumerate(rows):
            y = 9 - i
            x = 0
            for char in row:
                if char.isdigit():
                    x += int(char)
                elif char.isalpha():
                    piece = Piece.from_symbol(self._convert_symbol(char))
                    if piece:
                        self._board[y][x] = piece
                    x += 1
    
    @staticmethod
    def _convert_symbol(symbol: str) -> str:
        convert_map = {
            'k': 'K', 'a': 'A', 'b': 'B', 'n': 'N', 'r': 'R', 'c': 'C', 'p': 'P',
            'K': 'k', 'A': 'a', 'B': 'b', 'N': 'n', 'R': 'r', 'C': 'c', 'P': 'p'
        }
        return convert_map.get(symbol, symbol)
    
    def get_piece(self, pos: Position) -> Optional[Piece]:
        if not self.is_valid_position(pos):
            return None
        return self._board[pos.y][pos.x]
    
    def set_piece(self, pos: Position, piece: Optional[Piece]) -> None:
        if self.is_valid_position(pos):
            self._board[pos.y][pos.x] = piece
    
    @staticmethod
    def is_valid_position(pos: Position) -> bool:
        return 0 <= pos.x < 9 and 0 <= pos.y < 10
    
    def can_move_to(self, pos: Position, color: Color) -> bool:
        if not self.is_valid_position(pos):
            return False
        
        piece = self.get_piece(pos)
        return piece is None or piece.color != color
    
    @lru_cache(maxsize=1000)
    def get_legal_moves(self, color: Color) -> List[Move]:
        moves = []
        for y in range(10):
            for x in range(9):
                pos = Position(x, y)
                piece = self.get_piece(pos)
                if piece and piece.color == color:
                    strategy = self._move_strategies.get(piece.type)
                    if strategy:
                        moves.extend(strategy.get_moves(self, pos))
        return moves
    
    def make_move(self, move: Move) -> ChessBoard:
        new_board = ChessBoard()
        new_board._board = self._board.copy()
        
        piece = new_board.get_piece(move.from_pos)
        new_board.set_piece(move.to_pos, piece)
        new_board.set_piece(move.from_pos, None)
        
        return new_board.flip()
    
    def flip(self) -> ChessBoard:
        new_board = ChessBoard()
        for y in range(10):
            for x in range(9):
                piece = self._board[y][x]
                if piece:
                    flipped_piece = Piece(
                        piece.type,
                        Color.RED if piece.color == Color.BLACK else Color.BLACK
                    )
                    new_board._board[9-y][8-x] = flipped_piece
        return new_board
    
    def to_state(self) -> str:
        rows = []
        for i in range(10):
            y = 9 - i
            row_parts = []
            empty_count = 0
            
            for x in range(9):
                piece = self._board[y][x]
                if piece:
                    if empty_count > 0:
                        row_parts.append(str(empty_count))
                        empty_count = 0
                    row_parts.append(piece.symbol)
                else:
                    empty_count += 1
            
            if empty_count > 0:
                row_parts.append(str(empty_count))
            
            rows.append(''.join(row_parts))
        
        return '/'.join(rows)
    
    def evaluate(self) -> float:
        red_value = black_value = 0
        
        for y in range(10):
            for x in range(9):
                piece = self._board[y][x]
                if piece:
                    if piece.color == Color.RED:
                        red_value += piece.value
                    else:
                        black_value += piece.value
        
        total = red_value + black_value
        if total == 0:
            return 0.0
        
        return np.tanh(((red_value - black_value) / total) * 3)
    
    def get_game_result(self) -> Tuple[GameResult, Optional[Move]]:
        red_king = black_king = None
        
        for y in range(10):
            for x in range(9):
                piece = self._board[y][x]
                if piece and piece.type == PieceType.KING:
                    if piece.color == Color.RED:
                        red_king = Position(x, y)
                    else:
                        black_king = Position(x, y)
        
        if not red_king:
            return GameResult.BLACK_WIN, None
        if not black_king:
            return GameResult.RED_WIN, None
        
        if red_king.x == black_king.x:
            blocked = False
            for y in range(red_king.y + 1, black_king.y):
                if self._board[y][red_king.x] is not None:
                    blocked = True
                    break
            if not blocked:
                return GameResult.RED_WIN, None
        
        black_moves = self.get_legal_moves(Color.BLACK)
        for move in black_moves:
            if move.to_pos == red_king:
                return GameResult.RED_WIN, move
        
        return GameResult.ONGOING, None

@dataclass
class GameState:
    board: ChessBoard
    current_player: Color = Color.BLACK
    move_history: List[Move] = field(default_factory=list)
    state_history: List[str] = field(default_factory=list)
    
    def make_move(self, move: Move) -> GameState:
        new_board = self.board.make_move(move)
        new_history = self.move_history + [move]
        new_state_history = self.state_history + [self.board.to_state()]
        new_player = Color.RED if self.current_player == Color.BLACK else Color.BLACK
        
        return GameState(new_board, new_player, new_history, new_state_history)
    
    def get_legal_moves(self) -> List[Move]:
        return self.board.get_legal_moves(self.current_player)
    
    def is_game_over(self) -> bool:
        result, _ = self.board.get_game_result()
        return result != GameResult.ONGOING
    
    def get_winner(self) -> Optional[Color]:
        result, _ = self.board.get_game_result()
        if result == GameResult.RED_WIN:
            return Color.RED
        elif result == GameResult.BLACK_WIN:
            return Color.BLACK
        return None

def done(state: str, turns: int = -1, need_check: bool = False):
    board = ChessBoard(state)
    result, final_move = board.get_game_result()
    
    if result != GameResult.ONGOING:
        v = 1 if result == GameResult.RED_WIN else -1
        move_str = str(final_move) if final_move else None
        return (True, v, move_str) if not need_check else (True, v, move_str, False)
    
    if need_check:
        return (False, 0, None, False)
    
    return (False, 0, None)

def step(state: str, action: str) -> str:
    board = ChessBoard(state)
    move = Move(
        Position(int(action[0]), int(action[1])),
        Position(int(action[2]), int(action[3]))
    )
    new_board = board.make_move(move)
    return new_board.to_state()

def get_legal_moves(state: str, board=None) -> List[str]:
    chess_board = ChessBoard(state)
    moves = chess_board.get_legal_moves(Color.BLACK)
    return [str(move) for move in moves]

def evaluate(state: str) -> float:
    board = ChessBoard(state)
    return board.evaluate()
