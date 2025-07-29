import numpy as np
from typing import Dict, Optional
from sources.chess.static_env import Position, Piece, PieceType, Color

class ZobristHash:
    """高效的Zobrist哈希系统，支持快速计算和增量更新"""
    
    def __init__(self, seed: int = 42):
        """初始化Zobrist哈希表"""
        np.random.seed(seed)
        
        # 为每个位置每种棋子预生成随机64位数
        self.hash_table: Dict[tuple, int] = {}
        
        # 棋子映射：符号 -> 索引
        self.piece_to_index = {
            'R': 0, 'r': 1,   # 车
            'N': 2, 'n': 3,   # 马  
            'B': 4, 'b': 5,   # 象
            'A': 6, 'a': 7,   # 士
            'K': 8, 'k': 9,   # 将
            'C': 10, 'c': 11, # 炮
            'P': 12, 'p': 13  # 兵
        }
        
        # 为每个位置每种棋子生成随机哈希值
        for row in range(10):
            for col in range(9):
                for piece_idx in range(14):
                    # 使用位运算确保在64位范围内
                    hash_val = (np.random.randint(0, 2**32) << 32) | np.random.randint(0, 2**32)
                    self.hash_table[(row, col, piece_idx)] = hash_val
        
        # 额外的游戏状态哈希值
        self.side_to_move_hash = (np.random.randint(0, 2**32) << 32) | np.random.randint(0, 2**32)
        
    def compute_hash_from_state(self, state: str) -> int:
        """从FEN状态字符串计算哈希值 - 优化版本"""
        hash_val = 0
        rows = state.split('/')
        
        # 预先计算位置映射，避免重复计算
        y = 9
        for row in rows:
            x = 0
            for char in row:
                if '1' <= char <= '9':
                    x += int(char)  # 跳过空位
                else:
                    # 直接使用字典查找，避免多次调用
                    piece_idx = self.piece_to_index.get(char)
                    if piece_idx is not None:
                        # 直接计算哈希表键，避免元组创建
                        hash_val ^= self.hash_table[(y, x, piece_idx)]
                    x += 1
            y -= 1
        
        return hash_val
    
    def compute_hash_from_board(self, board_2d) -> int:
        """从2D棋盘数组计算哈希值"""
        hash_val = 0
        
        for y in range(10):
            for x in range(9):
                piece_char = board_2d[y][x]
                if piece_char != '.' and piece_char in self.piece_to_index:
                    piece_idx = self.piece_to_index[piece_char]
                    hash_val ^= self.hash_table[(y, x, piece_idx)]
        
        return hash_val
    
    def update_hash_for_move(self, current_hash: int, from_pos: Position, 
                           to_pos: Position, moving_piece: str, 
                           captured_piece: Optional[str] = None) -> int:
        """增量更新哈希值 - 这是性能关键优化"""
        new_hash = current_hash
        
        # 移除起始位置的棋子
        if moving_piece in self.piece_to_index:
            piece_idx = self.piece_to_index[moving_piece]
            new_hash ^= self.hash_table[(from_pos.y, from_pos.x, piece_idx)]
        
        # 如果有吃子，移除目标位置的被吃棋子
        if captured_piece and captured_piece in self.piece_to_index:
            captured_idx = self.piece_to_index[captured_piece]
            new_hash ^= self.hash_table[(to_pos.y, to_pos.x, captured_idx)]
        
        # 添加目标位置的移动棋子
        if moving_piece in self.piece_to_index:
            piece_idx = self.piece_to_index[moving_piece]
            new_hash ^= self.hash_table[(to_pos.y, to_pos.x, piece_idx)]
        
        return new_hash
    
    def get_piece_hash(self, pos: Position, piece_char: str) -> int:
        """获取特定位置特定棋子的哈希值"""
        if piece_char not in self.piece_to_index:
            return 0
        
        piece_idx = self.piece_to_index[piece_char]
        return self.hash_table[(pos.y, pos.x, piece_idx)]
    
    def verify_hash_consistency(self, state: str, computed_hash: int) -> bool:
        """验证哈希值一致性（调试用）"""
        expected_hash = self.compute_hash_from_state(state)
        return expected_hash == computed_hash
    
    def get_hash_info(self) -> Dict:
        """获取哈希系统信息"""
        return {
            'total_entries': len(self.hash_table),
            'positions': 10 * 9,
            'piece_types': 14,
            'expected_entries': 10 * 9 * 14
        }

# 全局Zobrist哈希实例
_global_zobrist = None

def get_zobrist_hash() -> ZobristHash:
    """获取全局Zobrist哈希实例（单例模式）"""
    global _global_zobrist
    if _global_zobrist is None:
        _global_zobrist = ZobristHash()
    return _global_zobrist 