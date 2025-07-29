import os
import getpass
from sources.config import ModelConfig, TrainerConfig, GenerateDataConfig, ResourceConfig, trainsetting

class EnhancedPlayConfig:
    """增强的MCTS配置，包含所有优化参数"""
    def __init__(self):
        # 基础参数
        self.max_processes = 32
        self.search_threads = 64
        self.vram_frac = 1.0
        self.simulation_num_per_move = 3200
        self.thinking_loop = 1
        self.logging_thinking = True
        
        # UCB相关参数
        self.c_puct = 1.5
        self.noise_eps = 0.15
        self.dirichlet_alpha = 0.2
        self.tau_decay_rate = 0.9
        
        # 虚拟损失参数
        self.virtual_loss = 3
        self.virtual_loss_adaptive = True  # 启用自适应虚拟损失
        self.virtual_loss_multiplier = 1.5  # 动态虚拟损失倍数
        
        # 认输参数
        self.resign_threshold = -0.98
        self.min_resign_turn = 40
        self.enable_resign_rate = 0.5
        self.max_game_length = 200
        
        # 信息共享
        self.share_mtcs_info_in_self_play = False
        self.reset_mtcs_info_per_game = 5
        
        # 新增：RAVE相关参数
        self.enable_rave = True
        self.rave_k_value = 2000  # RAVE权重衰减参数
        self.rave_threshold = 50  # RAVE生效的最小访问次数
        
        # 新增：UCB1-TUNED参数
        self.enable_ucb_tuned = True
        self.variance_bound = 0.25  # UCB1-TUNED方差上界
        
        # 新增：转置表参数
        self.enable_transposition_table = True
        self.transposition_table_size = 1000000  # 转置表大小
        self.transposition_table_threshold = 10  # 使用缓存的最小访问次数
        
        # 新增：渐进解锁参数
        self.enable_progressive_unlock = True
        self.unlock_complexity_multiplier = 50  # 解锁复杂度倍数
        self.unlock_bonus = 0.1  # 解锁奖励值
        
        # 新增：搜索树复用
        self.enable_tree_reuse = True
        self.tree_reuse_depth = 3  # 树复用的最大深度
        
        # 新增：并行优化
        self.batch_size_neural_network = 32  # 神经网络批处理大小
        self.max_queue_size = 256  # 最大队列大小
        
        # 新增：搜索优化
        self.enable_smart_pruning = True  # 智能剪枝
        self.pruning_threshold = 0.01  # 剪枝阈值
        self.enable_early_stopping = True  # 早停
        self.early_stopping_confidence = 0.95  # 早停置信度

class EnhancedConfig:
    """增强的整体配置"""
    def __init__(self):
        # 保持原有配置
        self.trainsetting = trainsetting()
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.trainer = TrainerConfig()
        
        # 使用增强的Play配置
        self.play = EnhancedPlayConfig()
        self.play_data = GenerateDataConfig()

class MCTSOptimizationConfig:
    """MCTS优化专用配置"""
    def __init__(self):
        # 核心算法开关
        self.algorithms = {
            'ucb1_tuned': True,      # UCB1-TUNED算法
            'rave': True,            # RAVE算法  
            'progressive_unlock': True,  # 渐进解锁
            'virtual_loss_adaptive': True,  # 自适应虚拟损失
            'transposition_table': True,    # 转置表
            'tree_reuse': True,      # 搜索树复用
            'smart_pruning': True,   # 智能剪枝
            'parallel_optimization': True,  # 并行优化
        }
        
        # 性能参数
        self.performance = {
            'target_nodes_per_second': 100000,  # 目标搜索节点/秒
            'memory_limit_mb': 4096,  # 内存限制
            'cpu_threads': 64,        # CPU线程数
            'gpu_batch_size': 32,     # GPU批处理大小
        }
        
        # 调试参数
        self.debugging = {
            'enable_profiling': False,   # 启用性能分析
            'log_search_tree': False,    # 记录搜索树
            'export_search_stats': True, # 导出搜索统计
            'benchmark_mode': False,     # 基准测试模式
        }

# 预定义的优化级别
class OptimizationLevels:
    @staticmethod
    def conservative():
        """保守优化：启用安全的优化"""
        config = EnhancedPlayConfig()
        config.enable_rave = True
        config.enable_ucb_tuned = True
        config.enable_transposition_table = True
        config.enable_progressive_unlock = False  # 关闭可能不稳定的功能
        config.virtual_loss_adaptive = False
        return config
    
    @staticmethod
    def aggressive():
        """激进优化：启用所有优化"""
        config = EnhancedPlayConfig()
        # 所有优化都启用（默认值）
        return config
    
    @staticmethod
    def memory_optimized():
        """内存优化：减少内存使用"""
        config = EnhancedPlayConfig()
        config.transposition_table_size = 100000  # 减小转置表
        config.simulation_num_per_move = 1600     # 减少搜索次数
        config.search_threads = 32                # 减少线程数
        return config
    
    @staticmethod
    def speed_optimized():
        """速度优化：最大化搜索速度"""
        config = EnhancedPlayConfig()
        config.simulation_num_per_move = 6400     # 增加搜索次数
        config.search_threads = 128               # 增加线程数
        config.batch_size_neural_network = 64    # 增大批处理
        return config

# 性能基准测试配置
class BenchmarkConfig:
    """性能基准测试配置"""
    def __init__(self):
        self.test_positions = [
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",  # 开局
            "r1ba1ab1r/4k4/2n1c1n2/p1p1p1p1p/9/9/P1P1P1P1P/2N1C1N2/4K4/R1BA1AB1R",  # 中局
            "3k5/9/4b4/9/4P4/9/9/4C4/4K4/9",  # 残局
        ]
        
        self.metrics = {
            'nodes_per_second': [],
            'memory_usage_mb': [],
            'search_depth': [],
            'time_per_move': [],
            'cache_hit_rate': [],
        }
        
        self.test_duration_seconds = 60
        self.warmup_moves = 5

def create_benchmark_report(config, results):
    """创建性能基准测试报告"""
    report = f"""
# MCTS优化性能报告

## 配置信息
- UCB1-TUNED: {'启用' if config.enable_ucb_tuned else '禁用'}
- RAVE算法: {'启用' if config.enable_rave else '禁用'}
- 转置表: {'启用' if config.enable_transposition_table else '禁用'}
- 渐进解锁: {'启用' if config.enable_progressive_unlock else '禁用'}

## 性能指标
- 搜索节点/秒: {results.get('nodes_per_second', 0):.0f}
- 内存使用: {results.get('memory_usage_mb', 0):.1f} MB
- 平均搜索深度: {results.get('search_depth', 0):.1f}
- 每步用时: {results.get('time_per_move', 0):.2f} 秒
- 缓存命中率: {results.get('cache_hit_rate', 0):.1%}

## 相比基础版本提升
- 搜索效率: +{results.get('efficiency_improvement', 0):.0%}
- 内存效率: +{results.get('memory_efficiency', 0):.0%}
- 整体性能: +{results.get('overall_improvement', 0):.0%}
"""
    return report 