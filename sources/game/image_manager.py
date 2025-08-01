"""
图片资源管理器
提供统一的图片加载、缓存和管理功能
"""

import os
import json
import pygame
from typing import Dict, List, Optional, Tuple, Any
from logging import getLogger
from enum import Enum

from sources.chess.chessman import Rook, Knight, Cannon, King, Elephant, Mandarin, Pawn

logger = getLogger(__name__)


class PieceState(Enum):
    """棋子状态枚举"""
    NORMAL = 0
    SELECTED = 1


class ImageManager:
    """图片资源管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_dir = os.path.join(self.base_dir, 'images')
        self.config_file = os.path.join(self.base_dir, 'images_config.json')
        
        # 图片缓存
        self._image_cache: Dict[str, pygame.Surface] = {}
        
        # 配置数据
        self._config: Dict[str, Any] = {}
        self._current_theme = "default"
        
        # 棋子类型映射
        self._piece_type_mapping = {
            Rook: "rook",
            Knight: "knight", 
            Cannon: "cannon",
            King: "king",
            Elephant: "elephant",
            Mandarin: "mandarin",
            Pawn: "pawn"
        }
        
        # 支持的图片格式
        self._supported_formats = ['.gif', '.png', '.jpg', '.jpeg', '.bmp']
        
        self._load_config()
    
    def _load_config(self):
        """加载图片配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info(f"Loaded image config from {self.config_file}")
            else:
                # 创建默认配置
                self._create_default_config()
                logger.info("Created default image config")
        except Exception as e:
            logger.error(f"Failed to load image config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self._config = {
            "themes": {
                "default": {
                    "board": "Board.GIF",
                    "pieces": {
                        "red": {
                            "rook": ["RR.gif", "RRS.gif"],
                            "knight": ["RN.gif", "RNS.gif"],
                            "cannon": ["RC.gif", "RCS.gif"],
                            "king": ["RK.gif", "RKS.gif"],
                            "elephant": ["RB.gif", "RBS.gif"],
                            "mandarin": ["RA.gif", "RAS.gif"],
                            "pawn": ["RP.gif", "RPS.gif"]
                        },
                        "black": {
                            "rook": ["BR.gif", "BRS.gif"],
                            "knight": ["BN.gif", "BNS.gif"],
                            "cannon": ["BC.gif", "BCS.gif"],
                            "king": ["BK.gif", "BKS.gif"],
                            "elephant": ["BB.gif", "BBS.gif"],
                            "mandarin": ["BA.gif", "BAS.gif"],
                            "pawn": ["BP.gif", "BPS.gif"]
                        }
                    }
                }
            },
            "default_theme": "default"
        }
        
        # 保存默认配置
        self._save_config()
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save image config: {e}")
    
    def _find_image_file(self, filename: str, sub_dir: Optional[str] = None) -> Optional[str]:
        """查找图片文件，支持多种格式"""
        if sub_dir:
            base_path = os.path.join(self.images_dir, sub_dir)
        else:
            base_path = self.images_dir
            
        # 如果文件名已包含扩展名，直接查找
        if os.path.splitext(filename)[1]:
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                return full_path
        else:
            # 尝试不同的扩展名
            base_name = os.path.splitext(filename)[0]
            for ext in self._supported_formats:
                full_path = os.path.join(base_path, base_name + ext)
                if os.path.exists(full_path):
                    return full_path
        
        return None
    
    def _load_image(self, file_path: str) -> Optional[pygame.Surface]:
        """加载单个图片文件"""
        try:
            surface = pygame.image.load(file_path)
            return surface.convert_alpha()
        except pygame.error as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            return None
    
    def get_board_image(self, theme: Optional[str] = None) -> Optional[pygame.Surface]:
        """获取棋盘背景图片"""
        theme = theme or self._current_theme
        
        try:
            board_filename = self._config["themes"][theme]["board"]
            cache_key = f"board_{theme}"
            
            if cache_key not in self._image_cache:
                file_path = self._find_image_file(board_filename)
                if file_path:
                    image = self._load_image(file_path)
                    if image:
                        self._image_cache[cache_key] = image
                        logger.debug(f"Loaded board image: {file_path}")
                    else:
                        return None
                else:
                    logger.error(f"Board image not found: {board_filename}")
                    return None
            
            return self._image_cache[cache_key]
            
        except KeyError as e:
            logger.error(f"Board image config not found for theme {theme}: {e}")
            return None
    
    def get_piece_images(self, piece_type: type, is_red: bool, 
                        theme: Optional[str] = None) -> Optional[List[pygame.Surface]]:
        """获取棋子图片列表 [普通状态, 选中状态]"""
        theme = theme or self._current_theme
        
        try:
            # 获取棋子类型名称
            type_name = self._piece_type_mapping.get(piece_type)
            if not type_name:
                logger.error(f"Unknown piece type: {piece_type}")
                return None
            
            # 获取颜色
            color = "red" if is_red else "black"
            
            # 获取图片文件名列表
            filenames = self._config["themes"][theme]["pieces"][color][type_name]
            
            cache_key = f"piece_{theme}_{color}_{type_name}"
            
            if cache_key not in self._image_cache:
                images = []
                for filename in filenames:
                    file_path = self._find_image_file(filename, "Piece")
                    if file_path:
                        image = self._load_image(file_path)
                        if image:
                            images.append(image)
                        else:
                            logger.error(f"Failed to load piece image: {filename}")
                            return None
                    else:
                        logger.error(f"Piece image not found: {filename}")
                        return None
                
                if images:
                    self._image_cache[cache_key] = images
                    logger.debug(f"Loaded piece images: {color} {type_name}")
                else:
                    return None
            
            return self._image_cache[cache_key]
            
        except KeyError as e:
            logger.error(f"Piece image config not found: {e}")
            return None
    
    def get_scaled_piece_images(self, piece_type: type, is_red: bool, 
                               width: int, height: int,
                               theme: Optional[str] = None) -> Optional[List[pygame.Surface]]:
        """获取缩放后的棋子图片"""
        images = self.get_piece_images(piece_type, is_red, theme)
        if not images:
            return None
        
        cache_key = f"scaled_{theme or self._current_theme}_{type(piece_type).__name__}_{is_red}_{width}x{height}"
        
        if cache_key not in self._image_cache:
            scaled_images = []
            for image in images:
                scaled_image = pygame.transform.scale(image, (width, height))
                scaled_images.append(scaled_image)
            self._image_cache[cache_key] = scaled_images
        
        return self._image_cache[cache_key]
    
    def set_theme(self, theme: str):
        """切换主题"""
        if theme in self._config["themes"]:
            self._current_theme = theme
            logger.info(f"Switched to theme: {theme}")
        else:
            logger.error(f"Theme not found: {theme}")
    
    def get_available_themes(self) -> List[str]:
        """获取可用主题列表"""
        return list(self._config["themes"].keys())
    
    def clear_cache(self):
        """清空图片缓存"""
        self._image_cache.clear()
        logger.info("Image cache cleared")
    
    def preload_theme(self, theme: str):
        """预加载指定主题的所有图片"""
        logger.info(f"Preloading theme: {theme}")
        
        # 预加载棋盘
        self.get_board_image(theme)
        
        # 预加载所有棋子
        for piece_type in self._piece_type_mapping.keys():
            self.get_piece_images(piece_type, True, theme)   # 红方
            self.get_piece_images(piece_type, False, theme)  # 黑方
        
        logger.info(f"Theme {theme} preloaded successfully")


# 全局图片管理器实例
image_manager = ImageManager()
