import gc
import psutil
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from logging import getLogger

logger = getLogger(__name__)

class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_history = []
        self.monitor_start_time = time.time()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': self.process.memory_percent(),   # 内存使用百分比
        }
    
    def record_memory_usage(self, label: str = ""):
        """记录当前内存使用"""
        current_memory = self.get_memory_usage()
        timestamp = time.time() - self.monitor_start_time
        
        self.memory_history.append({
            'timestamp': timestamp,
            'label': label,
            'memory': current_memory
        })
        
        if current_memory['rss_mb'] > self.peak_memory['rss_mb']:
            self.peak_memory = current_memory
            
        return current_memory
    
    def get_memory_growth(self) -> float:
        """获取内存增长量（MB）"""
        current = self.get_memory_usage()
        return current['rss_mb'] - self.initial_memory['rss_mb']
    
    def get_memory_report(self) -> str:
        """生成内存使用报告"""
        current = self.get_memory_usage()
        growth = self.get_memory_growth()
        
        report = f"""
📊 内存使用报告
{'='*50}
🚀 初始内存: {self.initial_memory['rss_mb']:.1f} MB
📈 当前内存: {current['rss_mb']:.1f} MB  
⬆️  内存增长: {growth:+.1f} MB
🔝 峰值内存: {self.peak_memory['rss_mb']:.1f} MB
📊 内存占用: {current['percent']:.1f}%
⏱️  监控时长: {time.time() - self.monitor_start_time:.1f}秒

💡 建议:
"""
        
        if growth > 100:
            report += "⚠️  内存增长较大，建议启用内存清理\n"
        if current['percent'] > 80:
            report += "🔴 内存使用率过高，需要立即清理\n"
        elif current['percent'] > 60:
            report += "🟡 内存使用率较高，建议适当清理\n"
        else:
            report += "✅ 内存使用正常\n"
            
        return report

class SearchTreeManager:
    """搜索树内存管理器"""
    
    def __init__(self, ai_player):
        self.ai_player = ai_player
        self.cleanup_threshold = 50000  # 节点数阈值
        self.cleanup_ratio = 0.3       # 清理比例
        self.min_visit_threshold = 5   # 最小访问次数
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300    # 清理间隔（秒）
        
    def get_tree_stats(self) -> Dict[str, Any]:
        """获取搜索树统计信息"""
        if not hasattr(self.ai_player, 'tree'):
            return {}
            
        tree = self.ai_player.tree
        total_nodes = len(tree)
        
        if total_nodes == 0:
            return {'total_nodes': 0}
        
        # 统计访问次数分布
        visit_counts = []
        total_visits = 0
        max_visits = 0
        
        for state, node in tree.items():
            visits = getattr(node, 'sum_n', 0)
            visit_counts.append(visits)
            total_visits += visits
            max_visits = max(max_visits, visits)
        
        # 计算统计数据
        avg_visits = total_visits / total_nodes if total_nodes > 0 else 0
        visit_counts.sort()
        median_visits = visit_counts[total_nodes // 2] if total_nodes > 0 else 0
        
        # 计算低访问节点数量
        low_visit_nodes = sum(1 for v in visit_counts if v < self.min_visit_threshold)
        
        return {
            'total_nodes': total_nodes,
            'total_visits': total_visits,
            'avg_visits': avg_visits,
            'median_visits': median_visits,
            'max_visits': max_visits,
            'low_visit_nodes': low_visit_nodes,
            'low_visit_ratio': low_visit_nodes / total_nodes if total_nodes > 0 else 0
        }
    
    def should_cleanup(self) -> bool:
        """判断是否需要清理"""
        stats = self.get_tree_stats()
        current_time = time.time()
        
        # 基于节点数量的清理
        if stats.get('total_nodes', 0) > self.cleanup_threshold:
            return True
            
        # 基于时间间隔的清理
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            return True
            
        # 基于低访问节点比例的清理
        if stats.get('low_visit_ratio', 0) > 0.6:
            return True
            
        return False
    
    def cleanup_search_tree(self, force: bool = False) -> Dict[str, Any]:
        """清理搜索树中的低价值节点"""
        if not hasattr(self.ai_player, 'tree'):
            return {'status': 'no_tree'}
            
        if not force and not self.should_cleanup():
            return {'status': 'no_cleanup_needed'}
        
        tree = self.ai_player.tree
        initial_count = len(tree)
        
        if initial_count == 0:
            return {'status': 'empty_tree'}
        
        # 收集所有节点的访问信息
        node_info = []
        for state, node in tree.items():
            visits = getattr(node, 'sum_n', 0)
            last_updated = getattr(node, 'last_updated', 0)
            node_info.append((state, visits, last_updated))
        
        # 按访问次数排序，保留高访问节点
        node_info.sort(key=lambda x: x[1], reverse=True)
        
        # 计算要保留的节点数量
        keep_count = max(
            int(initial_count * (1 - self.cleanup_ratio)),
            1000  # 至少保留1000个节点
        )
        
        # 清理低访问节点
        nodes_to_remove = node_info[keep_count:]
        removed_count = 0
        
        for state, visits, _ in nodes_to_remove:
            if visits < self.min_visit_threshold:
                try:
                    del tree[state]
                    removed_count += 1
                except KeyError:
                    pass
        
        # 更新清理时间
        self.last_cleanup_time = time.time()
        
        # 强制垃圾回收
        gc.collect()
        
        result = {
            'status': 'cleaned',
            'initial_nodes': initial_count,
            'removed_nodes': removed_count,
            'remaining_nodes': len(tree),
            'cleanup_ratio': removed_count / initial_count if initial_count > 0 else 0
        }
        
        logger.info(f"搜索树清理完成: {removed_count}/{initial_count} 节点被移除")
        return result
    
    def cleanup_caches(self) -> Dict[str, Any]:
        """清理各种缓存"""
        cleanup_results = {}
        
        # 清理哈希缓存
        if hasattr(self.ai_player, 'hash_cache'):
            initial_hash_cache = len(self.ai_player.hash_cache)
            # 保留最近使用的一半
            if initial_hash_cache > 1000:
                keys_to_remove = list(self.ai_player.hash_cache.keys())[initial_hash_cache//2:]
                for key in keys_to_remove:
                    del self.ai_player.hash_cache[key]
                cleanup_results['hash_cache'] = {
                    'initial': initial_hash_cache,
                    'removed': len(keys_to_remove),
                    'remaining': len(self.ai_player.hash_cache)
                }
        
        # 清理转置表
        if hasattr(self.ai_player, 'transposition_table') and self.ai_player.transposition_table:
            tt = self.ai_player.transposition_table
            if hasattr(tt, 'table'):
                initial_tt_size = len(tt.table)
                # 简单清理：移除低访问节点
                if initial_tt_size > tt.max_size * 0.8:
                    min_visits = 10
                    keys_to_remove = [k for k, v in tt.table.items() if getattr(v, 'sum_n', 0) < min_visits]
                    for key in keys_to_remove[:len(keys_to_remove)//2]:
                        del tt.table[key]
                    cleanup_results['transposition_table'] = {
                        'initial': initial_tt_size,
                        'removed': len(keys_to_remove),
                        'remaining': len(tt.table)
                    }
        
        return cleanup_results

class MemoryManager:
    """综合内存管理器"""
    
    def __init__(self, ai_player):
        self.ai_player = ai_player
        self.monitor = MemoryMonitor()
        self.tree_manager = SearchTreeManager(ai_player)
        
        # 内存管理配置
        self.memory_limit_mb = 2048      # 内存限制 (2GB)
        self.warning_threshold = 0.8     # 警告阈值
        self.critical_threshold = 0.9    # 紧急清理阈值
        
        # 自动清理配置
        self.auto_cleanup_enabled = True
        self.last_memory_check = time.time()
        self.memory_check_interval = 30  # 每30秒检查一次内存
        
    def check_memory_health(self) -> Dict[str, Any]:
        """检查内存健康状况"""
        current_memory = self.monitor.get_memory_usage()
        tree_stats = self.tree_manager.get_tree_stats()
        
        # 计算内存压力
        memory_pressure = current_memory['rss_mb'] / self.memory_limit_mb
        
        health_status = 'healthy'
        if memory_pressure > self.critical_threshold:
            health_status = 'critical'
        elif memory_pressure > self.warning_threshold:
            health_status = 'warning'
        
        return {
            'status': health_status,
            'memory_pressure': memory_pressure,
            'current_memory_mb': current_memory['rss_mb'],
            'memory_limit_mb': self.memory_limit_mb,
            'tree_nodes': tree_stats.get('total_nodes', 0),
            'memory_growth_mb': self.monitor.get_memory_growth(),
            'recommendations': self._get_recommendations(health_status, memory_pressure, tree_stats)
        }
    
    def _get_recommendations(self, health_status: str, memory_pressure: float, 
                           tree_stats: Dict) -> List[str]:
        """获取内存优化建议"""
        recommendations = []
        
        if health_status == 'critical':
            recommendations.append("🔴 立即执行内存清理")
            recommendations.append("🔴 考虑重启AI以释放内存")
        elif health_status == 'warning':
            recommendations.append("🟡 建议清理搜索树")
            recommendations.append("🟡 清理缓存数据")
        
        if tree_stats.get('total_nodes', 0) > 50000:
            recommendations.append("🌳 搜索树过大，建议清理")
            
        if tree_stats.get('low_visit_ratio', 0) > 0.5:
            recommendations.append("🧹 存在大量低访问节点，建议清理")
            
        return recommendations
    
    def auto_memory_management(self) -> Optional[Dict[str, Any]]:
        """自动内存管理"""
        if not self.auto_cleanup_enabled:
            return None
            
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return None
            
        self.last_memory_check = current_time
        health = self.check_memory_health()
        
        actions_taken = []
        
        if health['status'] == 'critical':
            # 紧急清理
            cleanup_result = self.tree_manager.cleanup_search_tree(force=True)
            actions_taken.append(f"紧急清理搜索树: {cleanup_result}")
            
            cache_cleanup = self.tree_manager.cleanup_caches()
            actions_taken.append(f"清理缓存: {cache_cleanup}")
            
            # 强制垃圾回收
            gc.collect()
            actions_taken.append("执行垃圾回收")
            
        elif health['status'] == 'warning':
            # 预防性清理
            if self.tree_manager.should_cleanup():
                cleanup_result = self.tree_manager.cleanup_search_tree()
                actions_taken.append(f"预防性清理: {cleanup_result}")
        
        if actions_taken:
            logger.info(f"自动内存管理: {actions_taken}")
            return {
                'triggered': True,
                'health': health,
                'actions': actions_taken,
                'memory_after': self.monitor.get_memory_usage()
            }
        
        return None
    
    def get_memory_summary(self) -> str:
        """获取内存使用总结"""
        health = self.check_memory_health()
        memory_report = self.monitor.get_memory_report()
        tree_stats = self.tree_manager.get_tree_stats()
        
        summary = f"""
🧠 AI内存管理总结
{'='*50}
{memory_report}

🌳 搜索树状态:
- 总节点数: {tree_stats.get('total_nodes', 0):,}
- 平均访问: {tree_stats.get('avg_visits', 0):.1f}
- 低访问节点: {tree_stats.get('low_visit_nodes', 0):,} ({tree_stats.get('low_visit_ratio', 0)*100:.1f}%)

🎯 健康状态: {health['status'].upper()}
💾 内存压力: {health['memory_pressure']*100:.1f}%

📋 建议:
"""
        
        for rec in health['recommendations']:
            summary += f"  {rec}\n"
            
        return summary 