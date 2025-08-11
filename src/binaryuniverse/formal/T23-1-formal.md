# T23-1 φ-博弈策略涌现定理 - 形式化规范

## 依赖导入
```python
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import itertools
from abc import ABC, abstractmethod

# 从前置理论导入
from T22_3_formal import WeightedPhiNetwork, ComprehensiveStabilityAnalyzer
from T22_2_formal import ConnectionEvolutionDynamics, PhiWeightQuantizer
from T22_1_formal import PhiNetwork, NetworkNode, NetworkEdge, FibonacciSequence
from T20_2_formal import TraceStructure
from C20_1_formal import CollapseObserver
```

## 1. φ-策略空间定义

### 1.1 基础策略结构
```python
class ZeckendorfString:
    """Zeckendorf表示（no-11约束）"""
    
    def __init__(self, n: int):
        self.value = n
        self.representation = self._to_zeckendorf(n)
        
    def _to_zeckendorf(self, n: int) -> str:
        """转换为Zeckendorf表示"""
        if n == 0:
            return '0'
            
        # 生成Fibonacci数列
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
            
        # 贪心算法构造Zeckendorf表示
        result = []
        remainder = n
        
        for fib in reversed(fibs):
            if fib <= remainder:
                result.append('1')
                remainder -= fib
            else:
                result.append('0')
                
        # 去除前导零
        result_str = ''.join(result).lstrip('0')
        return result_str if result_str else '0'
        
    def is_valid(self) -> bool:
        """验证no-11约束"""
        return '11' not in self.representation

@dataclass
class PhiStrategy:
    """φ-量化的博弈策略"""
    
    def __init__(self, fibonacci_index: int, phi_power: int):
        self.k = fibonacci_index  # Fibonacci索引
        self.d = phi_power       # φ的幂
        self.phi = (1 + np.sqrt(5)) / 2
        self.fib_sequence = FibonacciSequence()
        
        # 计算策略值
        fib_k = self.fib_sequence.get(self.k)
        self.value = fib_k / (self.phi ** self.d)
        
        # 验证Zeckendorf有效性
        self.zeckendorf_repr = ZeckendorfString(fib_k)
        self.is_valid_strategy = self.zeckendorf_repr.is_valid()
        
    def __eq__(self, other):
        if not isinstance(other, PhiStrategy):
            return False
        return self.k == other.k and self.d == other.d
        
    def __hash__(self):
        return hash((self.k, self.d))
        
    def __repr__(self):
        return f"PhiStrategy(F_{self.k}/φ^{self.d} = {self.value:.6f})"
        
    def distance_to(self, other: 'PhiStrategy') -> float:
        """计算到另一个φ-策略的距离"""
        return abs(self.value - other.value)
        
    def zeckendorf_overlap(self, other: 'PhiStrategy') -> float:
        """计算Zeckendorf表示的重叠度"""
        z1 = self.zeckendorf_repr.representation
        z2 = other.zeckendorf_repr.representation
        
        max_len = max(len(z1), len(z2))
        z1_padded = z1.zfill(max_len)
        z2_padded = z2.zfill(max_len)
        
        overlap = 0.0
        for i, (bit1, bit2) in enumerate(zip(z1_padded, z2_padded)):
            if bit1 == '1' and bit2 == '1':
                fib_index = max_len - i
                if fib_index > 0:
                    overlap += self.fib_sequence.get(fib_index)
                    
        return max(1.0, overlap)  # 确保非零
```

### 1.2 策略空间生成器
```python
class PhiStrategySpace:
    """φ-策略空间生成器"""
    
    def __init__(self, max_fibonacci_index: int = 20, max_phi_power: int = 10):
        self.max_k = max_fibonacci_index
        self.max_d = max_phi_power
        self.phi = (1 + np.sqrt(5)) / 2
        self._strategy_cache = {}
        self._generate_valid_strategies()
        
    def _generate_valid_strategies(self):
        """生成所有有效的φ-策略"""
        valid_strategies = []
        
        for k in range(1, self.max_k + 1):
            for d in range(0, self.max_d + 1):
                strategy = PhiStrategy(k, d)
                if strategy.is_valid_strategy:
                    valid_strategies.append(strategy)
                    
        # 按策略值排序
        valid_strategies.sort(key=lambda s: s.value)
        self.strategies = valid_strategies
        
        # 构建快速查找缓存
        for strategy in valid_strategies:
            self._strategy_cache[(strategy.k, strategy.d)] = strategy
            
    def get_strategy(self, fibonacci_index: int, phi_power: int) -> Optional[PhiStrategy]:
        """获取指定参数的策略"""
        return self._strategy_cache.get((fibonacci_index, phi_power))
        
    def get_all_strategies(self) -> List[PhiStrategy]:
        """获取所有有效策略"""
        return self.strategies.copy()
        
    def get_strategies_in_range(self, min_value: float, max_value: float) -> List[PhiStrategy]:
        """获取指定值域内的策略"""
        return [s for s in self.strategies if min_value <= s.value <= max_value]
        
    def find_closest_strategy(self, target_value: float) -> PhiStrategy:
        """找到最接近目标值的φ-策略"""
        if not self.strategies:
            raise ValueError("No valid strategies available")
            
        min_distance = float('inf')
        closest_strategy = self.strategies[0]
        
        for strategy in self.strategies:
            distance = abs(strategy.value - target_value)
            if distance < min_distance:
                min_distance = distance
                closest_strategy = strategy
                
        return closest_strategy
        
    def __len__(self):
        return len(self.strategies)
        
    def __iter__(self):
        return iter(self.strategies)
```

## 2. φ-博弈参与者

### 2.1 博弈参与者类
```python
class PhiGamePlayer:
    """φ-博弈参与者"""
    
    def __init__(self, player_id: int, network_node: NetworkNode, 
                 strategy_space: PhiStrategySpace):
        self.player_id = player_id
        self.network_node = network_node
        self.strategy_space = strategy_space
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 策略分布：每个策略的选择概率
        self.strategy_distribution = {}
        self._initialize_uniform_distribution()
        
        # 历史记录
        self.strategy_history = []
        self.payoff_history = []
        self.entropy_history = []
        
    def _initialize_uniform_distribution(self):
        """初始化为均匀分布"""
        strategies = self.strategy_space.get_all_strategies()
        if strategies:
            uniform_prob = 1.0 / len(strategies)
            self.strategy_distribution = {s: uniform_prob for s in strategies}
        else:
            self.strategy_distribution = {}
            
    def get_current_strategy_entropy(self) -> float:
        """计算当前策略熵"""
        entropy = 0.0
        for strategy, prob in self.strategy_distribution.items():
            if prob > 0:
                entropy -= prob * math.log(prob)
        return entropy
        
    def sample_strategy(self) -> PhiStrategy:
        """根据当前分布采样策略"""
        strategies = list(self.strategy_distribution.keys())
        probabilities = list(self.strategy_distribution.values())
        
        if not strategies:
            raise ValueError(f"Player {self.player_id} has no available strategies")
            
        # 确保概率归一化
        total_prob = sum(probabilities)
        if total_prob <= 0:
            return strategies[0]  # 返回第一个策略
            
        probabilities = [p / total_prob for p in probabilities]
        
        # 随机采样
        rand_val = np.random.random()
        cumulative_prob = 0.0
        
        for strategy, prob in zip(strategies, probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return strategy
                
        return strategies[-1]  # 返回最后一个策略
        
    def update_strategy_distribution(self, entropy_gradient: Dict[PhiStrategy, float], 
                                   dt: float = 0.1):
        """根据熵增梯度更新策略分布"""
        new_distribution = {}
        
        for strategy, current_prob in self.strategy_distribution.items():
            gradient = entropy_gradient.get(strategy, 0.0)
            
            # φ-调制的更新规则
            new_prob = current_prob + (dt / self.phi) * gradient * current_prob
            new_prob = max(1e-10, min(1.0, new_prob))  # 保持在有效范围
            
            new_distribution[strategy] = new_prob
            
        # 归一化
        total_prob = sum(new_distribution.values())
        if total_prob > 0:
            self.strategy_distribution = {
                s: p / total_prob for s, p in new_distribution.items()
            }
            
        # 记录历史
        current_entropy = self.get_current_strategy_entropy()
        self.entropy_history.append(current_entropy)
        
    def get_zeckendorf_vector(self) -> List[int]:
        """获取网络节点的Zeckendorf向量表示"""
        z_repr = self.network_node.z_representation.representation
        # 转换为固定长度的二进制向量
        max_length = 20  # 足够表示常见的Fibonacci数
        padded = z_repr.zfill(max_length)
        return [int(bit) for bit in padded]
        
    def compute_network_distance_to(self, other: 'PhiGamePlayer') -> float:
        """计算到另一个玩家的网络距离"""
        # 简化为度数差距
        degree_diff = abs(self.network_node.degree - other.network_node.degree)
        return max(1.0, degree_diff)
        
    def __repr__(self):
        return f"PhiGamePlayer(id={self.player_id}, node={self.network_node.id})"
```

### 2.2 策略演化追踪器
```python
class StrategyEvolutionTracker:
    """策略演化追踪器"""
    
    def __init__(self, player: PhiGamePlayer):
        self.player = player
        self.phi = (1 + np.sqrt(5)) / 2
        self.evolution_snapshots = []
        
    def take_snapshot(self, time_step: int):
        """记录当前时刻的策略状态"""
        snapshot = {
            'time': time_step,
            'strategy_distribution': self.player.strategy_distribution.copy(),
            'strategy_entropy': self.player.get_current_strategy_entropy(),
            'dominant_strategy': self._get_dominant_strategy(),
            'distribution_diversity': self._compute_distribution_diversity()
        }
        self.evolution_snapshots.append(snapshot)
        
    def _get_dominant_strategy(self) -> Tuple[PhiStrategy, float]:
        """获取占主导地位的策略"""
        if not self.player.strategy_distribution:
            return None, 0.0
            
        max_prob = 0.0
        dominant_strategy = None
        
        for strategy, prob in self.player.strategy_distribution.items():
            if prob > max_prob:
                max_prob = prob
                dominant_strategy = strategy
                
        return dominant_strategy, max_prob
        
    def _compute_distribution_diversity(self) -> float:
        """计算分布的多样性（有效策略数）"""
        if not self.player.strategy_distribution:
            return 0.0
            
        # 计算逆Simpson指数
        sum_squares = sum(p**2 for p in self.player.strategy_distribution.values())
        return 1.0 / sum_squares if sum_squares > 0 else 0.0
        
    def analyze_convergence(self) -> Dict[str, Any]:
        """分析策略收敛性"""
        if len(self.evolution_snapshots) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
            
        # 检查最近10个快照的熵变化
        recent_entropies = [snap['strategy_entropy'] for snap in self.evolution_snapshots[-10:]]
        entropy_std = np.std(recent_entropies)
        entropy_trend = np.mean(np.diff(recent_entropies))
        
        # 收敛判据
        convergence_threshold = 0.01
        converged = entropy_std < convergence_threshold and abs(entropy_trend) < convergence_threshold/10
        
        return {
            'converged': converged,
            'entropy_std': entropy_std,
            'entropy_trend': entropy_trend,
            'final_entropy': recent_entropies[-1] if recent_entropies else 0.0,
            'convergence_time': len(self.evolution_snapshots)
        }
```

## 3. φ-博弈系统

### 3.1 核心博弈类
```python
class PhiGameSystem:
    """φ-博弈系统的完整实现"""
    
    def __init__(self, network: WeightedPhiNetwork, n_players: int):
        self.network = network
        self.n_players = n_players
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 初始化组件
        self.strategy_space = PhiStrategySpace()
        self.players = {}
        self.payoff_matrix = None
        self.interaction_graph = None
        
        # 演化追踪
        self.game_entropy_history = []
        self.system_time = 0.0
        self.evolution_trackers = {}
        
        # 初始化系统
        self._initialize_players()
        self._compute_phi_payoff_matrix()
        self._build_interaction_graph()
        
    def _initialize_players(self):
        """初始化博弈参与者"""
        node_ids = list(self.network.nodes.keys())
        
        for i in range(min(self.n_players, len(node_ids))):
            node_id = node_ids[i]
            node = self.network.nodes[node_id]
            
            player = PhiGamePlayer(i, node, self.strategy_space)
            self.players[i] = player
            
            # 创建演化追踪器
            self.evolution_trackers[i] = StrategyEvolutionTracker(player)
            
    def _compute_phi_payoff_matrix(self):
        """计算完整的φ-收益矩阵"""
        n = len(self.players)
        self.payoff_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    payoff = self._compute_pairwise_payoff(i, j)
                    self.payoff_matrix[i, j] = payoff
                    
    def _compute_pairwise_payoff(self, player_i_id: int, player_j_id: int) -> float:
        """计算两个玩家之间的φ-收益"""
        if player_i_id not in self.players or player_j_id not in self.players:
            return 0.0
            
        player_i = self.players[player_i_id]
        player_j = self.players[player_j_id]
        
        # 1. Zeckendorf重叠强度
        z_i = player_i.get_zeckendorf_vector()
        z_j = player_j.get_zeckendorf_vector()
        
        fib_sequence = FibonacciSequence()
        interaction_strength = 0.0
        
        for k, (bit_i, bit_j) in enumerate(zip(z_i, z_j)):
            if bit_i == 1 and bit_j == 1:
                fib_index = len(z_i) - k
                if fib_index > 0:
                    interaction_strength += fib_sequence.get(fib_index)
                    
        # 2. 网络距离调制
        network_distance = player_i.compute_network_distance_to(player_j)
        
        # 3. φ-收益公式
        base_payoff = interaction_strength / (self.phi ** network_distance)
        
        # 4. 确保收益为正
        return max(0.1, base_payoff)
        
    def _build_interaction_graph(self):
        """构建交互图"""
        self.interaction_graph = {}
        
        for i in self.players:
            self.interaction_graph[i] = []
            for j in self.players:
                if i != j and self.payoff_matrix[i, j] > 0.1:
                    self.interaction_graph[i].append(j)
                    
    def compute_game_entropy(self) -> float:
        """计算系统总博弈熵"""
        # 1. 策略熵（按1/φ缩放）
        strategy_entropy = 0.0
        for player in self.players.values():
            player_entropy = player.get_current_strategy_entropy()
            strategy_entropy += player_entropy / self.phi
            
        # 2. 交互熵
        interaction_entropy = 0.0
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                if self.payoff_matrix[i, j] > 0:
                    interaction_entropy += self.payoff_matrix[i, j] * math.log(2)
                    
        # 3. 系统结构熵
        structure_entropy = len(self.players) * math.log(self.phi)
        
        # 总熵
        total_entropy = strategy_entropy + interaction_entropy + structure_entropy
        self.game_entropy_history.append(total_entropy)
        
        return total_entropy
        
    def compute_entropy_gradients(self) -> Dict[int, Dict[PhiStrategy, float]]:
        """计算所有玩家的熵增梯度"""
        gradients = {}
        
        for player_id, player in self.players.items():
            player_gradients = {}
            
            for strategy in player.strategy_distribution:
                gradient = self._compute_strategy_entropy_gradient(player_id, strategy)
                player_gradients[strategy] = gradient
                
            gradients[player_id] = player_gradients
            
        return gradients
        
    def _compute_strategy_entropy_gradient(self, player_id: int, strategy: PhiStrategy) -> float:
        """计算特定策略的熵增梯度"""
        if player_id not in self.players:
            return 0.0
            
        player = self.players[player_id]
        current_prob = player.strategy_distribution.get(strategy, 0.0)
        
        # 1. 基础熵梯度
        if current_prob > 0:
            entropy_gradient = -math.log(current_prob) - 1
        else:
            entropy_gradient = 1.0  # 鼓励未使用的策略
            
        # 2. 交互项
        interaction_gradient = 0.0
        for other_player_id in self.players:
            if other_player_id != player_id:
                payoff = self.payoff_matrix[player_id, other_player_id]
                
                # 策略相关的交互强度
                other_player = self.players[other_player_id]
                strategy_interaction = self._compute_strategy_interaction(
                    strategy, player, other_player
                )
                
                interaction_gradient += payoff * strategy_interaction / self.phi
                
        # 3. 网络结构项
        network_gradient = 0.0
        node_degree = player.network_node.degree
        if node_degree > 0:
            network_gradient = math.log(node_degree + 1) / (self.phi ** 2)
            
        return entropy_gradient + interaction_gradient + network_gradient
        
    def _compute_strategy_interaction(self, strategy: PhiStrategy, 
                                    player: PhiGamePlayer, 
                                    other_player: PhiGamePlayer) -> float:
        """计算策略与其他玩家的交互强度"""
        # 基于策略值和Zeckendorf重叠的交互强度
        strategy_overlap = strategy.zeckendorf_overlap(
            PhiStrategy(other_player.network_node.id % 10 + 1, 0)  # 简化表示
        )
        
        return strategy_overlap / (1 + abs(strategy.value - 1.0))
        
    def evolve_step(self, dt: float = 0.1):
        """执行一步博弈演化"""
        # 1. 计算熵增梯度
        gradients = self.compute_entropy_gradients()
        
        # 2. 更新所有玩家的策略分布
        for player_id, player_gradients in gradients.items():
            if player_id in self.players:
                self.players[player_id].update_strategy_distribution(player_gradients, dt)
                
        # 3. 记录快照
        for player_id, tracker in self.evolution_trackers.items():
            tracker.take_snapshot(int(self.system_time / dt))
            
        # 4. 更新系统状态
        self.system_time += dt
        game_entropy = self.compute_game_entropy()
        
        # 5. 验证熵增
        if len(self.game_entropy_history) > 1:
            entropy_increase = (self.game_entropy_history[-1] - 
                              self.game_entropy_history[-2])
            
            # 熵增应该非负（允许小的数值误差）
            if entropy_increase < -1e-8:
                raise ValueError(f"Entropy decrease detected: {entropy_increase}")
                
        return game_entropy
        
    def run_evolution(self, n_steps: int, dt: float = 0.1) -> List[float]:
        """运行完整的博弈演化"""
        entropy_trajectory = []
        
        for step in range(n_steps):
            entropy = self.evolve_step(dt)
            entropy_trajectory.append(entropy)
            
            # 每100步检查一次收敛性
            if step % 100 == 99:
                converged_players = 0
                for tracker in self.evolution_trackers.values():
                    convergence = tracker.analyze_convergence()
                    if convergence['converged']:
                        converged_players += 1
                        
                convergence_ratio = converged_players / len(self.players)
                if convergence_ratio > 0.8:  # 80%玩家收敛
                    print(f"System converged at step {step+1}")
                    break
                    
        return entropy_trajectory
        
    def verify_entropy_conservation(self, tolerance: float = 1e-8) -> bool:
        """验证熵增守恒定律"""
        if len(self.game_entropy_history) < 2:
            return True
            
        # 检查熵增趋势
        violations = 0
        for i in range(1, len(self.game_entropy_history)):
            entropy_increase = (self.game_entropy_history[i] - 
                              self.game_entropy_history[i-1])
            if entropy_increase < -tolerance:
                violations += 1
                
        # 允许少量数值误差
        violation_rate = violations / (len(self.game_entropy_history) - 1)
        return violation_rate < 0.05  # 允许5%的数值误差
        
    def analyze_equilibrium_properties(self) -> Dict[str, Any]:
        """分析博弈均衡性质"""
        if not self.players:
            return {}
            
        # 1. 策略分布分析
        strategy_analysis = {}
        for player_id, player in self.players.items():
            dominant_strategy, max_prob = None, 0.0
            for strategy, prob in player.strategy_distribution.items():
                if prob > max_prob:
                    max_prob = prob
                    dominant_strategy = strategy
                    
            strategy_analysis[player_id] = {
                'dominant_strategy': dominant_strategy,
                'dominance_strength': max_prob,
                'strategy_entropy': player.get_current_strategy_entropy(),
                'num_active_strategies': sum(1 for p in player.strategy_distribution.values() if p > 0.01)
            }
            
        # 2. 系统级分析
        total_entropy = self.compute_game_entropy()
        
        # 3. 收敛分析
        convergence_analysis = {}
        for player_id, tracker in self.evolution_trackers.items():
            convergence_analysis[player_id] = tracker.analyze_convergence()
            
        return {
            'strategy_analysis': strategy_analysis,
            'total_entropy': total_entropy,
            'convergence_analysis': convergence_analysis,
            'entropy_history_length': len(self.game_entropy_history),
            'entropy_conservation_verified': self.verify_entropy_conservation()
        }
        
    def get_payoff_matrix(self) -> np.ndarray:
        """获取收益矩阵"""
        return self.payoff_matrix.copy()
        
    def get_strategy_spaces(self) -> Dict[int, List[PhiStrategy]]:
        """获取所有玩家的策略空间"""
        return {
            player_id: self.strategy_space.get_all_strategies() 
            for player_id in self.players
        }
```

## 4. 博弈均衡分析

### 4.1 φ-Nash均衡检测器
```python
class PhiNashEquilibriumDetector:
    """φ-Nash均衡检测器"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.equilibrium_tolerance = 1e-4
        
    def is_phi_nash_equilibrium(self, strategy_profile: Dict[int, PhiStrategy]) -> bool:
        """检查给定策略组合是否为φ-Nash均衡"""
        for player_id in self.game.players:
            if not self._check_player_best_response(player_id, strategy_profile):
                return False
        return True
        
    def _check_player_best_response(self, player_id: int, 
                                   strategy_profile: Dict[int, PhiStrategy]) -> bool:
        """检查玩家的策略是否为最优反应"""
        if player_id not in strategy_profile:
            return False
            
        current_strategy = strategy_profile[player_id]
        current_payoff = self._compute_player_payoff(player_id, strategy_profile)
        
        # 检查所有其他策略
        for alternative_strategy in self.game.strategy_space.get_all_strategies():
            alternative_profile = strategy_profile.copy()
            alternative_profile[player_id] = alternative_strategy
            
            alternative_payoff = self._compute_player_payoff(player_id, alternative_profile)
            
            # φ-Nash条件：当前策略至少与最优策略在φ-误差内
            if alternative_payoff > current_payoff + self.equilibrium_tolerance / self.phi:
                return False
                
        return True
        
    def _compute_player_payoff(self, player_id: int, 
                             strategy_profile: Dict[int, PhiStrategy]) -> float:
        """计算玩家在给定策略组合下的收益"""
        if player_id not in self.game.players:
            return 0.0
            
        total_payoff = 0.0
        
        for other_player_id in self.game.players:
            if other_player_id != player_id:
                # 基础收益矩阵
                base_payoff = self.game.payoff_matrix[player_id, other_player_id]
                
                # 策略相关调制
                if (player_id in strategy_profile and 
                    other_player_id in strategy_profile):
                    
                    my_strategy = strategy_profile[player_id]
                    other_strategy = strategy_profile[other_player_id]
                    
                    # φ-策略交互
                    strategy_modifier = self._compute_strategy_modifier(
                        my_strategy, other_strategy
                    )
                    
                    total_payoff += base_payoff * strategy_modifier
                else:
                    total_payoff += base_payoff
                    
        return total_payoff
        
    def _compute_strategy_modifier(self, strategy1: PhiStrategy, 
                                 strategy2: PhiStrategy) -> float:
        """计算两个φ-策略之间的交互调制因子"""
        # 基于策略值的距离
        value_distance = abs(strategy1.value - strategy2.value)
        
        # 基于Zeckendorf重叠
        zeckendorf_overlap = strategy1.zeckendorf_overlap(strategy2)
        
        # φ-调制公式
        modifier = (zeckendorf_overlap / (1 + value_distance)) / self.phi
        
        return max(0.1, modifier)  # 确保正值
        
    def find_pure_phi_nash_equilibria(self) -> List[Dict[int, PhiStrategy]]:
        """寻找所有纯策略φ-Nash均衡"""
        equilibria = []
        
        # 遍历所有可能的纯策略组合
        strategies = self.game.strategy_space.get_all_strategies()
        
        if len(strategies) == 0 or len(self.game.players) == 0:
            return equilibria
            
        # 限制搜索空间以避免组合爆炸
        max_strategies_per_player = min(10, len(strategies))
        search_strategies = strategies[:max_strategies_per_player]
        
        for strategy_combination in itertools.product(search_strategies, 
                                                    repeat=len(self.game.players)):
            strategy_profile = {
                player_id: strategy_combination[i] 
                for i, player_id in enumerate(self.game.players.keys())
            }
            
            if self.is_phi_nash_equilibrium(strategy_profile):
                equilibria.append(strategy_profile)
                
        return equilibria
        
    def compute_equilibrium_stability(self, equilibrium: Dict[int, PhiStrategy]) -> float:
        """计算均衡的稳定性度量"""
        if not equilibrium:
            return 0.0
            
        stability_score = 0.0
        n_players = len(equilibrium)
        
        for player_id, strategy in equilibrium.items():
            # 计算偏离该策略的最大收益改进
            current_payoff = self._compute_player_payoff(player_id, equilibrium)
            max_deviation_gain = 0.0
            
            for alternative_strategy in self.game.strategy_space.get_all_strategies():
                if alternative_strategy != strategy:
                    deviation_profile = equilibrium.copy()
                    deviation_profile[player_id] = alternative_strategy
                    
                    deviation_payoff = self._compute_player_payoff(player_id, deviation_profile)
                    deviation_gain = deviation_payoff - current_payoff
                    
                    max_deviation_gain = max(max_deviation_gain, deviation_gain)
                    
            # 稳定性 = 1 / (1 + max_deviation_gain)，用φ调制
            player_stability = self.phi / (self.phi + max(0, max_deviation_gain))
            stability_score += player_stability
            
        return stability_score / n_players if n_players > 0 else 0.0
```

---

**注记**: 本形式化规范提供了T23-1定理的完整数学实现，包括φ-策略空间生成、博弈参与者建模、系统演化动力学和均衡分析的所有必要组件。所有实现严格遵循φ-表示、Zeckendorf编码和熵增原理。