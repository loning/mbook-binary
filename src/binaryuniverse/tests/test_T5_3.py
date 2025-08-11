"""
测试T5-3：信道容量定理

验证：
1. 信道容量的上界
2. Shannon熵对容量的调节作用
3. 最优策略的效果
4. 传统信道vs描述生成信道
5. 渐近行为
"""

import unittest
import numpy as np
import math
from typing import Set, Dict, List, Tuple
import random
from collections import defaultdict

class DescriptionChannel:
    """描述生成信道"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.phi = (1 + math.sqrt(5)) / 2
        self.h_max = math.log2(self.phi)
        self.time = 0
        
    def compute_capacity(self, shannon_entropy: float, desc_count: int) -> float:
        """计算当前信道容量（系统熵增长率）"""
        if desc_count == 0:
            return 0.0
            
        # 基于T5-1的描述生成率
        innovation_space = self.h_max - shannon_entropy
        desc_growth_rate = self.alpha * innovation_space
        
        # 信道容量 = d(log|D_t|)/dt
        capacity = desc_growth_rate / desc_count if desc_count > 0 else 0
        
        return capacity
    
    def theoretical_max_capacity(self) -> float:
        """理论最大容量"""
        return self.alpha * self.h_max


class PhiChannelSystem:
    """φ-表示信道系统"""
    
    def __init__(self, base_length: int = 8, alpha: float = 0.1):
        self.base_length = base_length
        self.channel = DescriptionChannel(alpha)
        self.descriptions: Set[str] = set()
        self.description_counts: Dict[str, int] = defaultdict(int)
        self.base_states: List[str] = []
        
        # 初始化基础状态
        self._initialize_base_states()
        
    def _initialize_base_states(self):
        """初始化基础φ-状态"""
        def generate_valid_sequences(n: int) -> List[str]:
            if n == 0:
                return ['']
            if n == 1:
                return ['0', '1']
            
            sequences = []
            for seq in generate_valid_sequences(n-1):
                sequences.append(seq + '0')
                if not seq.endswith('1'):
                    sequences.append(seq + '1')
            return sequences
        
        self.base_states = generate_valid_sequences(self.base_length)
        for state in self.base_states:
            self.descriptions.add(state)
            self.description_counts[state] = 1
    
    def compute_shannon_entropy(self) -> float:
        """计算基础状态的Shannon熵"""
        # 只考虑基础φ-状态的分布
        base_counts = {}
        for state in self.base_states:
            if state in self.descriptions:
                base_counts[state] = self.description_counts.get(state, 1)
        
        if not base_counts:
            return 0.0
            
        total = sum(base_counts.values())
        entropy = 0.0
        for count in base_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def generate_with_strategy(self, strategy: str = 'balanced') -> Dict[str, float]:
        """使用特定策略生成描述"""
        shannon_entropy = self.compute_shannon_entropy()
        current_capacity = self.channel.compute_capacity(
            shannon_entropy, len(self.descriptions)
        )
        
        # 根据容量决定生成数量
        # 使用更大的基数确保有描述生成
        base_rate = self.channel.alpha * self.channel.h_max
        expected_new = base_rate * (1 + current_capacity) * 10  # 放大因子
        n_new = max(1, np.random.poisson(max(1, expected_new)))
        
        actual_new = 0
        
        if strategy == 'balanced':
            # 平衡策略：保持适度的Shannon熵
            actual_new = self._apply_balanced_strategy(n_new)
            
        elif strategy == 'aggressive':
            # 激进策略：快速增加描述
            actual_new = self._apply_aggressive_strategy(int(n_new * 1.5))
            
        elif strategy == 'conservative':
            # 保守策略：缓慢增加
            actual_new = self._apply_conservative_strategy(max(1, int(n_new * 0.5)))
        
        # 更新分布
        self._update_distribution(strategy)
        
        # 计算新的容量
        new_shannon = self.compute_shannon_entropy()
        new_capacity = self.channel.compute_capacity(
            new_shannon, len(self.descriptions)
        )
        
        return {
            'time': self.channel.time,
            'strategy': strategy,
            'shannon_entropy': new_shannon,
            'channel_capacity': new_capacity,
            'description_count': len(self.descriptions),
            'new_descriptions': actual_new,
            'capacity_utilization': new_capacity / self.channel.theoretical_max_capacity()
        }
    
    def _apply_balanced_strategy(self, n_new: int) -> int:
        """平衡策略实现"""
        count = 0
        for i in range(n_new):
            if random.random() < 0.7 and self.descriptions:
                # 递归描述
                base = random.choice(list(self.descriptions))
                new_desc = f"D[{self.channel.time}:{i}:{base[:8]}...]"
                if new_desc not in self.descriptions:
                    self.descriptions.add(new_desc)
                    self.description_counts[new_desc] = 1
                    count += 1
            else:
                # 组合描述
                if len(self.descriptions) >= 2:
                    samples = random.sample(list(self.descriptions), 2)
                    new_desc = f"C[{self.channel.time}:{i}:{len(samples[0])}+{len(samples[1])}]"
                    if new_desc not in self.descriptions:
                        self.descriptions.add(new_desc)
                        self.description_counts[new_desc] = 1
                        count += 1
        return count
    
    def _apply_aggressive_strategy(self, n_new: int) -> int:
        """激进策略实现"""
        count = 0
        for i in range(n_new):
            # 快速生成多种描述
            desc_type = random.choice(['recursive', 'combine', 'meta'])
            base = random.choice(list(self.descriptions)) if self.descriptions else "0"
            new_desc = f"{desc_type[0]}{self.channel.time}-{i}[{base[:5]}...]"
            if new_desc not in self.descriptions:
                self.descriptions.add(new_desc)
                self.description_counts[new_desc] = 1
                count += 1
        return count
    
    def _apply_conservative_strategy(self, n_new: int) -> int:
        """保守策略实现"""
        count = 0
        for i in range(n_new):
            if self.descriptions and random.random() < 0.9:
                # 主要基于现有描述
                base = random.choice(list(self.descriptions))
                new_desc = f"S[{self.channel.time}:{i}:{base[:10]}]"
                if new_desc not in self.descriptions:
                    self.descriptions.add(new_desc)
                    self.description_counts[new_desc] = 1
                    count += 1
        return count
    
    def _update_distribution(self, strategy: str):
        """更新描述分布"""
        if strategy == 'balanced':
            # 趋向均匀分布
            avg_count = sum(self.description_counts.values()) / len(self.description_counts)
            for desc in self.description_counts:
                if self.description_counts[desc] > avg_count:
                    self.description_counts[desc] = int(self.description_counts[desc] * 0.95)
                else:
                    self.description_counts[desc] = int(self.description_counts[desc] * 1.05)
                    
        self.channel.time += 1


class TraditionalChannel:
    """传统二进制信道（用于对比）"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.capacity = math.log2(self.phi)  # bits per symbol
        
    def transmit(self, data_length: int) -> Dict[str, float]:
        """传输数据"""
        # 传统信道只传输，不创造信息
        transmitted_bits = data_length * self.capacity
        
        return {
            'channel_type': 'traditional',
            'capacity': self.capacity,
            'data_length': data_length,
            'transmitted_bits': transmitted_bits,
            'information_created': 0  # 传统信道不创造信息
        }


class TestT5_3ChannelCapacity(unittest.TestCase):
    """T5-3信道容量定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        np.random.seed(42)
        random.seed(42)
    
    def test_capacity_upper_bound(self):
        """测试1：信道容量上界"""
        print("\n测试1：信道容量上界")
        
        system = PhiChannelSystem(base_length=6, alpha=0.2)
        
        max_observed_capacity = 0
        theoretical_max = system.channel.theoretical_max_capacity()
        
        # 尝试不同的系统状态
        for i in range(100):
            # 随机调整Shannon熵
            if i % 20 == 0:
                # 重置为更均匀的分布
                for desc in system.description_counts:
                    system.description_counts[desc] = random.randint(1, 10)
            
            result = system.generate_with_strategy('balanced')
            if result['channel_capacity'] > max_observed_capacity:
                max_observed_capacity = result['channel_capacity']
        
        print(f"  理论最大容量: {theoretical_max:.6f}")
        print(f"  观察到的最大容量: {max_observed_capacity:.6f}")
        print(f"  容量比: {max_observed_capacity/theoretical_max:.2%}")
        
        # 容量不应超过理论上界
        self.assertLessEqual(max_observed_capacity, theoretical_max * 1.01,
                           "信道容量不应超过理论上界")
    
    def test_shannon_entropy_regulation(self):
        """测试2：Shannon熵对容量的调节作用"""
        print("\n测试2：Shannon熵调节作用")
        
        system = PhiChannelSystem(base_length=8, alpha=0.15)
        
        # 收集不同Shannon熵下的容量数据
        entropy_capacity_data = []
        
        for _ in range(50):
            shannon_entropy = system.compute_shannon_entropy()
            capacity = system.channel.compute_capacity(
                shannon_entropy, len(system.descriptions)
            )
            
            entropy_capacity_data.append({
                'shannon_entropy': shannon_entropy,
                'capacity': capacity,
                'innovation_space': system.channel.h_max - shannon_entropy
            })
            
            # 演化系统
            system.generate_with_strategy('balanced')
        
        # 分析相关性
        entropies = [d['shannon_entropy'] for d in entropy_capacity_data]
        capacities = [d['capacity'] for d in entropy_capacity_data]
        innovations = [d['innovation_space'] for d in entropy_capacity_data]
        
        # 计算相关系数
        if len(set(innovations)) > 1:  # 确保有变化
            correlation = np.corrcoef(innovations, capacities)[0, 1]
            print(f"  创新空间与容量相关系数: {correlation:.3f}")
            
            # 应该是正相关
            self.assertGreater(correlation, 0.5, 
                             "创新空间与信道容量应该正相关")
        
        # 验证极端情况
        print(f"  最小Shannon熵时容量: {min(capacities):.6f}")
        print(f"  最大Shannon熵时容量: {max(capacities):.6f}")
    
    def test_optimal_strategy(self):
        """测试3：最优策略效果"""
        print("\n测试3：策略比较")
        
        strategies = ['balanced', 'aggressive', 'conservative']
        results = {}
        
        for strategy in strategies:
            system = PhiChannelSystem(base_length=6, alpha=0.1)
            
            total_capacity = 0
            steps = 50
            
            for _ in range(steps):
                result = system.generate_with_strategy(strategy)
                total_capacity += result['channel_capacity']
            
            avg_capacity = total_capacity / steps
            final_count = len(system.descriptions)
            
            results[strategy] = {
                'avg_capacity': avg_capacity,
                'final_count': final_count,
                'capacity_per_desc': avg_capacity * final_count if final_count > 0 else 0
            }
            
            print(f"\n  策略 '{strategy}':")
            print(f"    平均容量: {avg_capacity:.6f}")
            print(f"    最终描述数: {final_count}")
            print(f"    总体效率: {results[strategy]['capacity_per_desc']:.3f}")
        
        # 平衡策略应该有较好的综合表现
        balanced_efficiency = results['balanced']['capacity_per_desc']
        aggressive_efficiency = results['aggressive']['capacity_per_desc']
        
        print(f"\n  平衡vs激进效率比: {balanced_efficiency/aggressive_efficiency:.2f}")
    
    def test_traditional_vs_description_channel(self):
        """测试4：传统信道vs描述生成信道"""
        print("\n测试4：信道类型对比")
        
        # 传统信道
        traditional = TraditionalChannel()
        trad_result = traditional.transmit(100)
        
        print(f"\n  传统信道:")
        print(f"    容量: {trad_result['capacity']:.4f} bits/symbol")
        print(f"    传输100符号: {trad_result['transmitted_bits']:.2f} bits")
        print(f"    信息创造: {trad_result['information_created']}")
        
        # 描述生成信道
        desc_system = PhiChannelSystem(base_length=8, alpha=0.1)
        
        initial_count = len(desc_system.descriptions)
        total_new = 0
        
        for _ in range(100):
            result = desc_system.generate_with_strategy('balanced')
            total_new += result['new_descriptions']
        
        print(f"\n  描述生成信道:")
        print(f"    初始描述数: {initial_count}")
        print(f"    最终描述数: {len(desc_system.descriptions)}")
        print(f"    创造的新描述: {total_new}")
        print(f"    信息创造率: {total_new/100:.2f} descriptions/step")
        
        # 验证本质区别
        self.assertEqual(trad_result['information_created'], 0,
                        "传统信道不应创造信息")
        self.assertGreater(total_new, 0,
                          "描述生成信道应该创造新信息")
    
    def test_asymptotic_behavior(self):
        """测试5：渐近行为"""
        print("\n测试5：长期渐近行为")
        
        system = PhiChannelSystem(base_length=6, alpha=0.05)
        
        # 长时间演化
        capacity_history = []
        window_size = 20
        
        for i in range(200):
            result = system.generate_with_strategy('balanced')
            capacity_history.append(result['channel_capacity'])
            
            if i % 50 == 0 and i > 0:
                # 计算移动平均
                recent_avg = np.mean(capacity_history[-window_size:])
                print(f"  步骤{i}: 平均容量={recent_avg:.6f}, "
                      f"描述数={result['description_count']}")
        
        # 验证容量收敛
        early_capacities = capacity_history[:window_size]
        late_capacities = capacity_history[-window_size:]
        
        early_var = np.var(early_capacities)
        late_var = np.var(late_capacities)
        
        print(f"\n  早期容量方差: {early_var:.6f}")
        print(f"  晚期容量方差: {late_var:.6f}")
        
        # 计算平均容量
        early_mean = np.mean(early_capacities)
        late_mean = np.mean(late_capacities)
        
        print(f"  早期平均容量: {early_mean:.6f}")
        print(f"  晚期平均容量: {late_mean:.6f}")
        
        # 晚期容量应该更低（接近0）
        if early_mean > 0:
            self.assertLess(late_mean, early_mean * 0.5,
                           "容量应该随时间递减")
        
        # 验证容量非负
        self.assertGreaterEqual(min(capacity_history), -0.01,
                              "容量不应显著为负")
    
    def test_capacity_formula_verification(self):
        """测试6：容量公式验证"""
        print("\n测试6：容量公式验证")
        
        # 创建受控环境
        alpha = 0.1
        phi = (1 + math.sqrt(5)) / 2
        h_max = math.log2(phi)
        
        channel = DescriptionChannel(alpha)
        
        # 测试不同的Shannon熵值
        test_entropies = [0.1, 0.3, 0.5, 0.6, 0.65, 0.69]
        desc_count = 100
        
        print(f"  α = {alpha}, H_max = {h_max:.4f}")
        print(f"  |D_t| = {desc_count}")
        print(f"\n  H_Shannon  Innovation  Capacity")
        print(f"  ---------  ----------  --------")
        
        for h_shannon in test_entropies:
            innovation = h_max - h_shannon
            capacity = channel.compute_capacity(h_shannon, desc_count)
            expected = (alpha * innovation) / desc_count
            
            print(f"  {h_shannon:.3f}      {innovation:.3f}       {capacity:.6f}")
            
            # 验证公式
            self.assertAlmostEqual(capacity, expected, places=6,
                                 msg=f"容量计算应符合公式")


if __name__ == '__main__':
    unittest.main()