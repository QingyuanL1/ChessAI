from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock, Condition
import concurrent.futures.thread
import hashlib
import math

import numpy as np
import sources.chess.static_env as senv
from sources.config_enhanced import EnhancedConfig as Config
from sources.chess.lookup_tables import Winner, ActionLabelsRed, flip_move
from time import time, sleep
import gc 
import sys

logger = getLogger(__name__)

class EnhancedVisitState:
    def __init__(self):
        self.a = defaultdict(EnhancedActionState)
        self.sum_n = 0
        self.visit = []
        self.p = None
        self.legal_moves = None
        self.waiting = False
        self.w = 0
        self.q_variance = 0  # 用于UCB1-TUNED
        self.last_updated = 0
        self.zobrist_hash = None

class EnhancedActionState:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0
        self.q_variance = 0
        self.rave_n = 0  # RAVE访问次数
        self.rave_w = 0  # RAVE总价值
        self.rave_q = 0  # RAVE平均价值
        self.unlock_threshold = 0  # 渐进解锁阈值

class TranspositionTable:
    """转置表，避免重复计算"""
    def __init__(self, max_size=1000000):
        self.table = {}
        self.max_size = max_size
        self.lock = Lock()
    
    def get(self, zobrist_hash):
        with self.lock:
            return self.table.get(zobrist_hash)
    
    def put(self, zobrist_hash, node):
        with self.lock:
            if len(self.table) >= self.max_size:
                # 简单的LRU：删除最少访问的节点
                min_visits = min(node.sum_n for node in self.table.values())
                keys_to_remove = [k for k, v in self.table.items() if v.sum_n == min_visits]
                for k in keys_to_remove[:len(keys_to_remove)//2]:
                    del self.table[k]
            
            self.table[zobrist_hash] = node

class Enhanced_AI_Player:
    def __init__(self, config: Config, search_tree=None, pipes=None, play_config=None, 
            enable_resign=False, debugging=False, uci=False, use_history=False, side=0):
        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = len(ActionLabelsRed)
        self.labels = ActionLabelsRed
        self.move_lookup = {move: i for move, i in zip(self.labels, range(self.labels_n))}
        self.pipe = pipes
        self.node_lock = defaultdict(Lock)
        self.use_history = use_history
        self.increase_temp = False

        if search_tree is None:
            self.tree = defaultdict(EnhancedVisitState)
        else:
            self.tree = search_tree

        self.root_state = None
        self.enable_resign = enable_resign
        self.debugging = debugging
        self.search_results = {}
        self.debug = {}
        self.side = side

        # 优化组件
        tt_size = self.play_config.transposition_table_size if hasattr(self.play_config, 'transposition_table_size') else 1000000
        self.transposition_table = TranspositionTable(max_size=tt_size) if hasattr(self.play_config, 'enable_transposition_table') and self.play_config.enable_transposition_table else None
        self.rave_table = defaultdict(lambda: defaultdict(list))
        self.zobrist_table = self._init_zobrist_table()
        
        # 线程管理
        self.s_lock = Lock()
        self.run_lock = Lock()
        self.q_lock = Lock()
        self.t_lock = Lock()
        self.buffer_planes = []
        self.buffer_history = []
        self.all_done = Lock()
        self.num_task = 0
        self.done_tasks = 0
        self.uci = uci
        self.no_act = None
        self.job_done = False

        self.executor = ThreadPoolExecutor(max_workers=self.play_config.search_threads + 2)
        self.executor.submit(self.receiver)
        self.executor.submit(self.sender)

    def _init_zobrist_table(self):
        """初始化Zobrist哈希表"""
        np.random.seed(42)  # 固定种子确保一致性
        table = {}
        # 为每个位置每种棋子生成随机数
        for row in range(10):
            for col in range(9):
                for piece_type in range(14):  # 14种不同的棋子状态
                    table[(row, col, piece_type)] = np.random.randint(0, 2**63)
        return table

    def compute_zobrist_hash(self, state):
        """计算局面的Zobrist哈希值"""
        hash_val = 0
        # 这里需要根据实际的state格式来实现
        # 简化版本，实际需要根据棋盘编码格式调整
        state_bytes = state.encode() if isinstance(state, str) else str(state).encode()
        return int(hashlib.md5(state_bytes).hexdigest()[:16], 16)

    def enhanced_select_action(self, state, is_root_node=False):
        """增强的UCB选择策略：UCB1-TUNED + RAVE"""
        node = self.tree[state]
        legal_moves = node.legal_moves
        
        if not legal_moves:
            return None
            
        best_score = -float('inf')
        best_action = None
        
        # RAVE权重计算
        rave_weight = self._compute_rave_weight(node.sum_n)
        
        for move in legal_moves:
            if is_root_node and self.no_act and move in self.no_act:
                continue
                
            action_state = node.a[move]
            
            # UCB1-TUNED分数
            ucb_score = self._compute_ucb_tuned(action_state, node)
            
            # RAVE分数
            rave_score = action_state.rave_q if action_state.rave_n > 0 else 0
            
            # 混合分数
            final_score = (1 - rave_weight) * ucb_score + rave_weight * rave_score
            
            # 渐进解锁奖励
            if self._should_unlock_move(move, node.sum_n):
                final_score += self._get_unlock_bonus(move)
            
            # 根节点添加噪声
            if is_root_node:
                noise = np.random.dirichlet([self.play_config.dirichlet_alpha] * len(legal_moves))[0]
                final_score = (1 - self.play_config.noise_eps) * final_score + \
                            self.play_config.noise_eps * noise
            
            if final_score > best_score:
                best_score = final_score
                best_action = move
        
        return best_action

    def _compute_ucb_tuned(self, action_state, parent_node):
        """计算UCB1-TUNED分数"""
        if action_state.n == 0:
            return float('inf')
        
        # 标准UCB项
        exploration_term = math.sqrt(2 * math.log(parent_node.sum_n) / action_state.n)
        
        # UCB1-TUNED的方差项
        variance_term = action_state.q_variance + math.sqrt(2 * math.log(parent_node.sum_n) / action_state.n)
        bound = self.play_config.variance_bound if hasattr(self.play_config, 'variance_bound') else 0.25
        variance_bound = min(bound, variance_term)  # 方差上界
        
        # 最终UCB1-TUNED分数
        ucb_tuned = action_state.q + exploration_term * variance_bound
        
        return ucb_tuned

    def _compute_rave_weight(self, n):
        """计算RAVE权重 beta = sqrt(k/(3n+k))"""
        k = self.play_config.rave_k_value if hasattr(self.play_config, 'rave_k_value') else 2000
        return math.sqrt(k / (3 * n + k))

    def _should_unlock_move(self, move, visit_count):
        """渐进解锁：复杂走法需要更多访问才解锁"""
        if not hasattr(self.play_config, 'enable_progressive_unlock') or not self.play_config.enable_progressive_unlock:
            return True
        complexity = self._get_move_complexity(move)
        multiplier = self.play_config.unlock_complexity_multiplier if hasattr(self.play_config, 'unlock_complexity_multiplier') else 50
        threshold = complexity * multiplier
        return visit_count >= threshold

    def _get_move_complexity(self, move):
        """评估走法复杂度"""
        # 简化实现：根据走法类型返回复杂度
        # 实际可以根据走法是否为杀棋、弃子、长将等判断
        return 1

    def _get_unlock_bonus(self, move):
        """获取解锁奖励"""
        return self.play_config.unlock_bonus if hasattr(self.play_config, 'unlock_bonus') else 0.1

    def enhanced_virtual_loss(self, action_state, parent_node):
        """动态虚拟损失"""
        base_loss = self.play_config.virtual_loss
        
        # 根据访问比例调整
        visit_ratio = action_state.n / max(1, parent_node.sum_n)
        dynamic_multiplier = 1 + visit_ratio * 0.5
        
        # 根据Q值调整：Q值越低，虚拟损失越大
        q_adjustment = max(0, 0.5 - action_state.q) * 2
        
        return base_loss * dynamic_multiplier + q_adjustment

    def MCTS_search_enhanced(self, state, history=[], is_root_node=False, real_hist=None):
        """增强的MCTS搜索"""
        zobrist_hash = self.compute_zobrist_hash(state)
        
        # 检查转置表
        if self.transposition_table:
            cached_node = self.transposition_table.get(zobrist_hash)
            threshold = self.play_config.transposition_table_threshold if hasattr(self.play_config, 'transposition_table_threshold') else 10
            if cached_node and cached_node.sum_n > threshold:
                return cached_node.q_variance
        
        while True:
            game_over, v, _ = senv.done(state)
            if game_over:
                v = v * 2
                self.executor.submit(self.enhanced_update_tree, None, v, history, zobrist_hash)
                break

            with self.node_lock[state]:
                if state not in self.tree:
                    # 扩展和评估
                    node = self.tree[state]
                    node.sum_n = 1
                    node.legal_moves = senv.get_legal_moves(state)
                    node.waiting = True
                    node.zobrist_hash = zobrist_hash
                    
                    # 存储到转置表
                    if self.transposition_table:
                        self.transposition_table.put(zobrist_hash, node)
                    
                    if is_root_node and real_hist:
                        self.expand_and_evaluate(state, history, real_hist)
                    else:
                        self.expand_and_evaluate(state, history)
                    break

                # 检查重复局面
                if state in history[:-1]:
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            if senv.will_check_or_catch(state, history[i+1]):
                                self.executor.submit(self.enhanced_update_tree, None, -1, history, zobrist_hash)
                            elif senv.be_catched(state, history[i+1]):
                                self.executor.submit(self.enhanced_update_tree, None, 1, history, zobrist_hash)
                            else:
                                self.executor.submit(self.enhanced_update_tree, None, 0, history, zobrist_hash)
                            break
                    break

                # 选择行动
                node = self.tree[state]
                if node.waiting:
                    node.visit.append(history)
                    break

                sel_action = self.enhanced_select_action(state, is_root_node)
                if sel_action is None:
                    break

                # 应用动态虚拟损失
                virtual_loss = self.enhanced_virtual_loss(node.a[sel_action], node)
                node.sum_n += 1
                
                action_state = node.a[sel_action]
                action_state.n += virtual_loss
                action_state.w -= virtual_loss
                action_state.q = action_state.w / action_state.n
                
                history.append(sel_action)
                state = senv.step(state, sel_action)
                history.append(state)

    def enhanced_update_tree(self, p, v, history, zobrist_hash=None):
        """增强的树更新，包含RAVE更新和方差计算"""
        state = history.pop()
        z = v

        if p is not None:
            with self.node_lock[state]:
                node = self.tree[state]
                node.p = p
                node.waiting = False
                if self.debugging:
                    self.debug[state] = (p, v)
                for hist in node.visit:
                    self.executor.submit(self.MCTS_search_enhanced, state, hist)
                node.visit = []

        # 回传更新，包含RAVE更新
        moves_in_path = []
        virtual_loss = self.play_config.virtual_loss
        
        while len(history) > 0:
            action = history.pop()
            state = history.pop()
            moves_in_path.append(action)
            v = -v
            
            with self.node_lock[state]:
                node = self.tree[state]
                action_state = node.a[action]
                
                # 标准更新
                old_q = action_state.q
                action_state.n += 1 - virtual_loss
                action_state.w += v + virtual_loss
                action_state.q = action_state.w / action_state.n
                
                # 更新方差（用于UCB1-TUNED）
                delta = action_state.q - old_q
                action_state.q_variance += delta * delta
                action_state.q_variance *= (action_state.n - 1) / action_state.n
                
                # RAVE更新：为后续路径中的所有走法更新RAVE值
                self._update_rave_values(node, moves_in_path[:-1], v)

        with self.t_lock:
            self.num_task -= 1
            if self.num_task <= 0:
                self.all_done.release()

    def _update_rave_values(self, node, future_moves, value):
        """更新RAVE值"""
        for move in future_moves:
            if move in node.a:
                action_state = node.a[move]
                action_state.rave_n += 1
                action_state.rave_w += value
                action_state.rave_q = action_state.rave_w / action_state.rave_n

    # 保持原有接口兼容性
    def action(self, state, turns, no_act=None, depth=None, infinite=False, hist=None, increase_temp=False):
        """主要action接口，使用增强的MCTS"""
        self.all_done.acquire(True)
        self.root_state = state
        self.no_act = no_act
        self.increase_temp = increase_temp
        
        if hist and len(hist) >= 5:
            hist = hist[-5:]
            
        done = 0
        if state in self.tree:
            done = self.tree[state].sum_n
            
        if no_act or increase_temp or done == self.play_config.simulation_num_per_move:
            done = 0
            
        self.done_tasks = done
        self.num_task = self.play_config.simulation_num_per_move - done
        
        if depth:
            self.num_task = depth - done if depth > done else 0
        if infinite:
            self.num_task = 100000
            
        start_time = time()
        
        # 使用增强的MCTS搜索
        if self.num_task > 0:
            all_tasks = self.num_task
            batch = all_tasks // self.config.play.search_threads
            if all_tasks % self.config.play.search_threads != 0:
                batch += 1
                
            for iter in range(batch):
                self.num_task = min(self.config.play.search_threads, 
                                  all_tasks - self.config.play.search_threads * iter)
                self.done_tasks += self.num_task
                
                for i in range(self.num_task):
                    self.executor.submit(self.MCTS_search_enhanced, state, [state], True, hist)
                    
                self.all_done.acquire(True)
                
        self.all_done.release()
        
        policy, resign = self.calc_policy(state, turns, no_act)
        
        if resign:
            return None, list(policy)
            
        if no_act is not None:
            for act in no_act:
                policy[self.move_lookup[act]] = 0
                
        my_action = int(np.random.choice(range(self.labels_n), 
                                       p=self.apply_temperature(policy, turns)))
        return self.labels[my_action], list(policy)

    # 保持其他必要的方法...
    def close(self, wait=True):
        self.job_done = True
        del self.tree
        gc.collect()
        if self.executor is not None:
            self.executor.shutdown(wait=wait)
    
    def sender(self):
        limit = 256
        while not self.job_done:
            self.run_lock.acquire()
            with self.q_lock:
                l = min(limit, len(self.buffer_history))
                if l > 0:
                    t_data = self.buffer_planes[0:l]
                    self.pipe.send(t_data)
                else:
                    self.run_lock.release()
                    sleep(0.001)

    def receiver(self):
        while not self.job_done:
            if self.pipe.poll(0.001):
                rets = self.pipe.recv()
            else:
                continue
            k = 0
            with self.q_lock:
                for ret in rets:
                    self.executor.submit(self.enhanced_update_tree, ret[0], ret[1], self.buffer_history[k])
                    k = k + 1
                self.buffer_planes = self.buffer_planes[k:]
                self.buffer_history = self.buffer_history[k:]
            self.run_lock.release()
    
    def expand_and_evaluate(self, state, history, real_hist=None):
        if self.use_history:
            if real_hist:
                state_planes = senv.state_history_to_planes(state, real_hist)
            else:
                state_planes = senv.state_history_to_planes(state, history)
        else:
            state_planes = senv.state_to_planes(state)
        with self.q_lock:
            self.buffer_planes.append(state_planes)
            self.buffer_history.append(history)
    
    def calc_policy(self, state, turns, no_act):
        node = self.tree[state]
        policy = np.zeros(self.labels_n)
        max_q_value = -100
        debug_result = {}

        for mov, action_state in node.a.items():
            policy[self.move_lookup[mov]] = action_state.n
            if no_act and mov in no_act:
                policy[self.move_lookup[mov]] = 0
                continue
            if self.debugging:
                debug_result[mov] = (action_state.n, action_state.q, action_state.p)
            if action_state.q > max_q_value:
                max_q_value = action_state.q

        if max_q_value < self.play_config.resign_threshold and self.enable_resign and turns > self.play_config.min_resign_turn:
            return policy, True

        if self.debugging:
            self.search_results = debug_result

        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum

        return policy, False
    
    def apply_temperature(self, policy, turn):
        tau = self.play_config.tau_decay_rate ** turn
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(len(policy))
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret = ret / np.sum(ret)
            return ret 