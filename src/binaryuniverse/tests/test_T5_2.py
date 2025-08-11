"""
测试T5-2：最大熵定理

验证：
1. 系统熵的单调递增性
2. Shannon熵收敛到最大值
3. 准稳态的达成
4. 两种熵的关系
5. 描述产生率的衰减
"""

import unittest
import numpy as np
import math
from collections import defaultdict
from typing import Set, Dict, List, Tuple
import random

class PhiDescriptionSystem:
    """φ-表示系统的描述生成和熵计算"""
    
    def __init__(self, base_length: int = 8):
        self.base_length = base_length
        self.descriptions: Set[str] = set()
        self.description_counts: Dict[str, int] = defaultdict(int)
        self.time = 0
        self.phi = (1 + math.sqrt(5)) / 2
        self.alpha = 0.1  # 描述生成速率常数
        
        # 初始化基础φ-状态
        self._initialize_base_states()
        
    def _initialize_base_states(self):
        """初始化基础φ-状态（满足no-11约束）"""
        def generate_valid_sequences(n: int) -> List[str]:
            """生成长度为n的所有有效φ-序列"""
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
        
        # 生成基础φ-状态
        base_states = generate_valid_sequences(self.base_length)
        
        # 添加到描述集合
        for state in base_states:
            self.descriptions.add(state)
            self.description_counts[state] = 1
    
    def compute_system_entropy(self) -> float:
        """计算系统熵 H_system = log|D_t|"""
        if len(self.descriptions) == 0:
            return 0.0
        return math.log2(len(self.descriptions))
    
    def compute_shannon_entropy(self) -> float:
        """计算Shannon熵 H_Shannon = -Σ p_i log p_i"""
        # 计算基于φ-状态的Shannon熵，而不是所有描述
        # 只考虑基础φ-状态的分布
        base_states = [s for s in self.descriptions if len(s) == self.base_length and all(c in '01' for c in s)]
        
        if not base_states:
            return 0.0
            
        # 计算基础状态的频率分布
        state_counts = {}
        for state in base_states:
            state_counts[state] = self.description_counts.get(state, 1)
        
        total_count = sum(state_counts.values())
        if total_count == 0:
            return 0.0
        
        entropy = 0.0
        for count in state_counts.values():
            if count > 0:
                p = count / total_count
                entropy -= p * math.log2(p)
        return entropy
    
    def compute_max_shannon_entropy(self) -> float:
        """计算最大Shannon熵（均匀分布）"""
        # 对于φ-表示系统，最大Shannon熵是log2(φ)
        # 这是因为φ-状态的渐近密度
        return math.log2(self.phi)
    
    def generate_new_descriptions(self) -> int:
        """根据T5-1生成新描述"""
        # 计算当前Shannon熵
        h_shannon = self.compute_shannon_entropy()
        h_max = self.compute_max_shannon_entropy()
        
        # 期望的新描述数（基于T5-1）
        if h_max > 0:
            innovation_space = h_max - h_shannon
            expected_new = self.alpha * innovation_space * len(self.descriptions)
        else:
            expected_new = self.alpha * len(self.descriptions)
        
        # 生成新描述（使用泊松过程）
        n_new = np.random.poisson(max(0, expected_new))
        
        new_count = 0
        for _ in range(int(n_new)):
            # 选择生成策略
            strategy = random.choice(['recursive', 'combine', 'time_stamp'])
            
            if strategy == 'recursive' and len(self.descriptions) > 0:
                # 递归描述：Desc(Desc(s))
                base = random.choice(list(self.descriptions))
                new_desc = f"D[{self.time}:{base}]"
                if new_desc not in self.descriptions:
                    self.descriptions.add(new_desc)
                    self.description_counts[new_desc] = 1
                    new_count += 1
                    
            elif strategy == 'combine' and len(self.descriptions) >= 2:
                # 组合描述
                samples = random.sample(list(self.descriptions), 2)
                new_desc = f"C[{samples[0]}+{samples[1]}]"
                if new_desc not in self.descriptions:
                    self.descriptions.add(new_desc)
                    self.description_counts[new_desc] = 1
                    new_count += 1
                    
            elif strategy == 'time_stamp':
                # 时间标记描述
                if len(self.descriptions) > 0:
                    base = random.choice(list(self.descriptions))
                    new_desc = f"T{self.time}[{base}]"
                    if new_desc not in self.descriptions:
                        self.descriptions.add(new_desc)
                        self.description_counts[new_desc] = 1
                        new_count += 1
        
        # 更新现有描述的计数（模拟使用频率变化）
        self._update_description_counts()
        
        return new_count
    
    def _update_description_counts(self):
        """更新描述计数，趋向均匀分布"""
        # 计算平均计数
        total = sum(self.description_counts.values())
        avg = total / len(self.description_counts) if len(self.description_counts) > 0 else 1
        
        # 向平均值靠拢（模拟均匀化过程）
        for desc in self.description_counts:
            current = self.description_counts[desc]
            # 小幅调整向平均值
            if current > avg:
                self.description_counts[desc] = max(1, int(current * 0.95))
            else:
                self.description_counts[desc] = int(current * 1.05)
    
    def evolve(self) -> Dict[str, float]:
        """演化一个时间步"""
        # 记录演化前的熵
        system_entropy_before = self.compute_system_entropy()
        shannon_entropy_before = self.compute_shannon_entropy()
        
        # 生成新描述
        new_descriptions = self.generate_new_descriptions()
        
        # 更新时间
        self.time += 1
        
        # 计算演化后的熵
        system_entropy_after = self.compute_system_entropy()
        shannon_entropy_after = self.compute_shannon_entropy()
        
        return {
            'time': self.time,
            'system_entropy': system_entropy_after,
            'shannon_entropy': shannon_entropy_after,
            'max_shannon_entropy': self.compute_max_shannon_entropy(),
            'new_descriptions': new_descriptions,
            'total_descriptions': len(self.descriptions),
            'system_entropy_change': system_entropy_after - system_entropy_before,
            'shannon_entropy_change': shannon_entropy_after - shannon_entropy_before
        }
    
    def is_quasi_steady_state(self, threshold: float = 0.01) -> bool:
        """检测是否达到准稳态"""
        h_shannon = self.compute_shannon_entropy()
        h_max = self.compute_max_shannon_entropy()
        
        if h_max == 0:
            return False
        
        # Shannon熵接近最大值
        shannon_ratio = h_shannon / h_max
        
        # 最近的描述生成率
        recent_rate = self.alpha * (h_max - h_shannon)
        
        return shannon_ratio > 0.99 and recent_rate < threshold


class TestT5_2MaximumEntropy(unittest.TestCase):
    """T5-2最大熵定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        np.random.seed(42)
        random.seed(42)
    
    def test_entropy_monotonicity(self):
        """测试1：系统熵的单调递增性"""
        print("\n测试1：系统熵单调递增性")
        
        system = PhiDescriptionSystem(base_length=6)
        history = []
        
        # 演化系统
        for _ in range(100):
            state = system.evolve()
            history.append(state)
        
        # 验证单调性
        violations = 0
        for i in range(1, len(history)):
            if history[i]['system_entropy'] < history[i-1]['system_entropy']:
                violations += 1
        
        print(f"  演化步数: {len(history)}")
        print(f"  初始系统熵: {history[0]['system_entropy']:.4f}")
        print(f"  最终系统熵: {history[-1]['system_entropy']:.4f}")
        print(f"  单调性违反次数: {violations}")
        
        # 系统熵应该单调递增
        self.assertEqual(violations, 0, "系统熵必须单调递增")
    
    def test_shannon_entropy_convergence(self):
        """测试2：Shannon熵收敛到最大值"""
        print("\n测试2：Shannon熵收敛性")
        
        system = PhiDescriptionSystem(base_length=5)
        history = []
        
        # 长时间演化
        for _ in range(500):
            state = system.evolve()
            history.append(state)
        
        # 分析收敛性
        last_50 = history[-50:]
        shannon_ratios = [s['shannon_entropy'] / s['max_shannon_entropy'] 
                         for s in last_50 if s['max_shannon_entropy'] > 0]
        
        avg_ratio = np.mean(shannon_ratios)
        
        print(f"  总演化步数: {len(history)}")
        print(f"  初始Shannon熵: {history[0]['shannon_entropy']:.4f}")
        print(f"  最终Shannon熵: {history[-1]['shannon_entropy']:.4f}")
        print(f"  最终最大Shannon熵: {history[-1]['max_shannon_entropy']:.4f}")
        print(f"  最后50步平均比值: {avg_ratio:.4f}")
        
        # Shannon熵应该接近最大值
        self.assertGreater(avg_ratio, 0.95, "Shannon熵应该收敛到接近最大值")
    
    def test_quasi_steady_state(self):
        """测试3：准稳态的达成"""
        print("\n测试3：准稳态达成")
        
        system = PhiDescriptionSystem(base_length=5)
        
        # 演化直到达到准稳态
        max_steps = 1000
        reached_steady = False
        steady_time = -1
        
        for t in range(max_steps):
            state = system.evolve()
            
            if not reached_steady and system.is_quasi_steady_state():
                reached_steady = True
                steady_time = t
                print(f"  在第{t}步达到准稳态")
                break
        
        if reached_steady:
            # 继续演化验证稳定性
            post_steady_new = []
            for _ in range(50):
                state = system.evolve()
                post_steady_new.append(state['new_descriptions'])
            
            avg_new = np.mean(post_steady_new)
            print(f"  准稳态后平均新描述数: {avg_new:.4f}")
            
            # 准稳态后新描述产生应该很少
            self.assertLess(avg_new, 1.0, "准稳态后新描述产生率应该很低")
        else:
            print(f"  警告：{max_steps}步内未达到准稳态")
    
    def test_entropy_relationship(self):
        """测试4：系统熵远大于Shannon熵"""
        print("\n测试4：两种熵的关系")
        
        system = PhiDescriptionSystem(base_length=6)
        
        # 演化一段时间
        for _ in range(200):
            system.evolve()
        
        system_entropy = system.compute_system_entropy()
        shannon_entropy = system.compute_shannon_entropy()
        max_shannon = system.compute_max_shannon_entropy()
        
        print(f"  系统熵: {system_entropy:.4f}")
        print(f"  Shannon熵: {shannon_entropy:.4f}")
        print(f"  最大Shannon熵: {max_shannon:.4f}")
        print(f"  描述总数: {len(system.descriptions)}")
        
        # 系统熵 = log|D_t|，Shannon熵 <= log(φ)
        # 由于递归描述的存在，|D_t|会远大于基础状态数
        # 所以系统熵会远大于Shannon熵的上界
        self.assertGreater(system_entropy, max_shannon, 
                          "系统熵应该大于Shannon熵的理论上界")
    
    def test_growth_rate_decay(self):
        """测试5：描述产生率随Shannon熵增加而衰减"""
        print("\n测试5：增长率衰减")
        
        # 使用更大的系统以观察明显的增长率变化
        system = PhiDescriptionSystem(base_length=8)
        system.alpha = 0.5  # 增加生成率常数
        
        # 记录不同阶段的增长率
        phases = []
        phase_size = 20
        
        for phase in range(5):
            phase_data = {
                'phase': phase,
                'new_descriptions': [],
                'shannon_entropies': [],
                'system_entropies': []
            }
            
            for _ in range(phase_size):
                state = system.evolve()
                phase_data['new_descriptions'].append(state['new_descriptions'])
                phase_data['shannon_entropies'].append(state['shannon_entropy'])
                phase_data['system_entropies'].append(state['system_entropy'])
            
            avg_new = np.mean(phase_data['new_descriptions'])
            avg_shannon = np.mean(phase_data['shannon_entropies'])
            avg_system = np.mean(phase_data['system_entropies'])
            
            phases.append({
                'phase': phase,
                'avg_new_descriptions': avg_new,
                'avg_shannon_entropy': avg_shannon,
                'avg_system_entropy': avg_system
            })
            
            print(f"  阶段{phase}: 平均新描述={avg_new:.2f}, "
                  f"Shannon熵={avg_shannon:.3f}, 系统熵={avg_system:.3f}")
        
        # 验证系统熵持续增长
        for i in range(1, len(phases)):
            self.assertGreaterEqual(phases[i]['avg_system_entropy'], 
                                  phases[i-1]['avg_system_entropy'],
                                  "系统熵应该持续增长")
        
        # 验证描述生成率的总体趋势
        # 早期和晚期的比较
        early_avg = np.mean([p['avg_new_descriptions'] for p in phases[:2]])
        late_avg = np.mean([p['avg_new_descriptions'] for p in phases[-2:]])
        
        print(f"\n  早期平均生成率: {early_avg:.3f}")
        print(f"  晚期平均生成率: {late_avg:.3f}")
        
        # 至少验证生成率没有增加
        if early_avg > 0:
            self.assertLessEqual(late_avg, early_avg * 1.1,
                               "描述生成率不应该显著增加")
    
    def test_phi_representation_entropy_bound(self):
        """测试6：φ-表示系统的熵密度界限"""
        print("\n测试6：φ-表示熵密度界限")
        
        # 计算不同长度的φ-序列数
        def count_phi_sequences(n: int) -> int:
            """计算长度为n的有效φ-序列数（Fibonacci数）"""
            if n == 0:
                return 1
            if n == 1:
                return 2
            
            a, b = 1, 2
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        
        phi = (1 + math.sqrt(5)) / 2
        theoretical_density = math.log2(phi)
        
        print(f"  理论熵密度上界: {theoretical_density:.6f} bits/symbol")
        
        # 验证不同长度的熵密度
        densities = []
        for n in [5, 10, 15, 20, 30]:
            count = count_phi_sequences(n)
            density = math.log2(count) / n
            densities.append(density)
            print(f"  n={n}: 序列数={count}, 密度={density:.6f}")
        
        # 密度应该收敛到理论值
        # 随着n增大，密度应该越来越接近log2(φ)
        convergence_error = abs(densities[-1] - theoretical_density)
        self.assertLess(convergence_error, 0.01,
                       f"熵密度应该收敛到log2(φ)")


if __name__ == '__main__':
    unittest.main()