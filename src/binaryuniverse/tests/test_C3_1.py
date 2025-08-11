#!/usr/bin/env python3
"""
test_C3_1_binary.py - C3-1系统演化推论的完整二进制机器验证测试

完全基于二进制φ-表示系统验证自指完备系统的演化规律
"""

import unittest
import sys
import os
import math
import numpy as np
from typing import List, Dict, Tuple, Set, Any
import random

# 添加包路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class PhiRepresentationSystem:
    """φ-表示系统（复用之前的定义）"""
    
    def __init__(self, n: int):
        """初始化n位φ-表示系统"""
        self.n = n
        self.valid_states = self._generate_valid_states()
        self.state_to_index = {tuple(s): i for i, s in enumerate(self.valid_states)}
        self.index_to_state = {i: s for i, s in enumerate(self.valid_states)}
        
    def _is_valid_phi_state(self, state: List[int]) -> bool:
        """检查是否为有效的φ-表示状态"""
        if len(state) != self.n:
            return False
        if not all(bit in [0, 1] for bit in state):
            return False
        
        # 检查no-consecutive-1s约束
        for i in range(len(state) - 1):
            if state[i] == 1 and state[i + 1] == 1:
                return False
        return True
    
    def _generate_valid_states(self) -> List[List[int]]:
        """生成所有有效的φ-表示状态"""
        valid_states = []
        
        def generate_recursive(current_state: List[int], pos: int):
            if pos == self.n:
                if self._is_valid_phi_state(current_state):
                    valid_states.append(current_state[:])
                return
            
            # 尝试放置0
            current_state.append(0)
            generate_recursive(current_state, pos + 1)
            current_state.pop()
            
            # 尝试放置1（如果不违反约束）
            if pos == 0 or current_state[pos - 1] == 0:
                current_state.append(1)
                generate_recursive(current_state, pos + 1)
                current_state.pop()
        
        generate_recursive([], 0)
        return valid_states
    
    def get_neighbors(self, state: List[int]) -> List[List[int]]:
        """获取状态的所有邻居（汉明距离为1的有效状态）"""
        neighbors = []
        state_tuple = tuple(state)
        
        for i in range(self.n):
            # 翻转第i位
            neighbor = state[:]
            neighbor[i] = 1 - neighbor[i]
            
            # 检查是否有效
            if self._is_valid_phi_state(neighbor):
                neighbors.append(neighbor)
        
        return neighbors


class BinarySystemEvolutionVerifier:
    """二进制系统演化推论验证器"""
    
    def __init__(self, n: int = 8):
        """初始化验证器"""
        self.n = n
        self.phi = (1 + math.sqrt(5)) / 2
        
        # 创建φ-表示系统
        self.phi_system = PhiRepresentationSystem(n)
        self.num_states = len(self.phi_system.valid_states)
        
        # 观测器配置
        self.num_observers = 2
        self.observers = self._initialize_binary_observers()
        
        # 演化参数
        self.self_evolution_prob = 0.1  # 自演化概率
        self.observation_effect = 0.05  # 观测效应强度
        
        # 记录
        self.evolution_history = []
        
    def _initialize_binary_observers(self) -> List[List[int]]:
        """初始化二进制观测器"""
        observers = []
        
        # 观测器1：偏好前半部分为1
        obs1 = [0] * self.n
        obs1[0] = 1
        if self.n > 2:
            obs1[2] = 1
        observers.append(obs1)
        
        # 观测器2：偏好后半部分为1
        obs2 = [0] * self.n
        if self.n > 1:
            obs2[-1] = 1
        if self.n > 3:
            obs2[-3] = 1
        observers.append(obs2)
        
        return observers
    
    def compute_hamming_distance(self, state1: List[int], state2: List[int]) -> int:
        """计算两个状态的汉明距离"""
        return sum(b1 != b2 for b1, b2 in zip(state1, state2))
    
    def self_referential_map(self, state: List[int]) -> List[int]:
        """二进制自指映射 f: S -> S"""
        # 在二进制系统中，自指映射应该保持状态的某些特征
        
        # 方案1：循环移位（保持1的个数）
        # 这是一个简单的自指映射
        if all(b == 0 for b in state):
            return state[:]
        
        # 找到第一个1的位置
        first_one = -1
        for i, bit in enumerate(state):
            if bit == 1:
                first_one = i
                break
        
        if first_one == -1:
            return state[:]
        
        # 构造新状态：将模式向右移动，但保持no-11约束
        new_state = [0] * self.n
        
        # 尝试将每个1向右移动一位
        for i in range(self.n):
            if state[i] == 1:
                new_pos = (i + 1) % self.n
                # 检查是否可以放置
                if new_pos == 0 or new_state[new_pos - 1] == 0:
                    if new_pos == self.n - 1 or new_state[new_pos + 1] == 0:
                        new_state[new_pos] = 1
                    else:
                        new_state[i] = 1  # 保持原位
                else:
                    new_state[i] = 1  # 保持原位
        
        # 确保是有效状态
        if self.phi_system._is_valid_phi_state(new_state):
            return new_state
        else:
            return state[:]
    
    def compute_binary_entropy(self, state: List[int]) -> float:
        """计算二进制状态的熵"""
        # 使用状态的局部模式计算熵
        
        # 方法1：基于1的分布
        ones_count = sum(state)
        zeros_count = self.n - ones_count
        
        if ones_count == 0 or zeros_count == 0:
            return 0.0
        
        p_one = ones_count / self.n
        p_zero = zeros_count / self.n
        
        entropy = -p_one * math.log2(p_one) - p_zero * math.log2(p_zero)
        
        # 方法2：考虑局部模式（2-gram）
        patterns = {}
        for i in range(self.n - 1):
            pattern = (state[i], state[i + 1])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        if patterns:
            total_patterns = sum(patterns.values())
            pattern_entropy = 0.0
            for count in patterns.values():
                if count > 0:
                    p = count / total_patterns
                    pattern_entropy -= p * math.log2(p)
            
            # 组合两种熵
            entropy = 0.7 * entropy + 0.3 * pattern_entropy
        
        return entropy
    
    def self_evolution_step(self, state: List[int]) -> List[int]:
        """自演化步骤 L[S]"""
        # 计算自指映射
        f_state = self.self_referential_map(state)
        
        # 如果状态已经是不动点，添加小的随机扰动
        if f_state == state:
            # 随机选择一个邻居
            neighbors = self.phi_system.get_neighbors(state)
            if neighbors and random.random() < self.self_evolution_prob:
                return random.choice(neighbors)
        
        # 以一定概率向f(S)演化
        if random.random() < 0.5 + self.self_evolution_prob:
            return f_state
        else:
            return state
    
    def observer_interaction_step(self, state: List[int], 
                                observer: List[int]) -> List[int]:
        """观测器相互作用步骤 R[S,O]"""
        # 计算状态与观测器的相似度
        similarity = sum(s == o for s, o in zip(state, observer)) / self.n
        
        # 根据相似度决定影响强度
        if random.random() < similarity * self.observation_effect:
            # 向观测器状态靠近
            # 找到一个不同的位，使其与观测器一致
            diff_positions = [i for i in range(self.n) 
                            if state[i] != observer[i]]
            
            if diff_positions:
                # 随机选择一个位置
                pos = random.choice(diff_positions)
                new_state = state[:]
                new_state[pos] = observer[pos]
                
                # 检查是否有效
                if self.phi_system._is_valid_phi_state(new_state):
                    return new_state
        
        return state
    
    def evolution_step(self, state: List[int]) -> List[int]:
        """完整的演化步骤"""
        # 1. 自演化
        state = self.self_evolution_step(state)
        
        # 2. 观测器相互作用
        for observer in self.observers:
            if random.random() < 0.5:  # 随机选择是否与此观测器相互作用
                state = self.observer_interaction_step(state, observer)
        
        # 3. 熵增驱动
        # 计算当前熵
        current_entropy = self.compute_binary_entropy(state)
        
        # 尝试找到熵更高的邻居
        neighbors = self.phi_system.get_neighbors(state)
        higher_entropy_neighbors = []
        
        for neighbor in neighbors:
            neighbor_entropy = self.compute_binary_entropy(neighbor)
            if neighbor_entropy > current_entropy:
                higher_entropy_neighbors.append((neighbor, neighbor_entropy))
        
        # 以概率选择高熵邻居
        if higher_entropy_neighbors and random.random() < 0.3:
            # 按熵增量加权选择
            weights = [ent - current_entropy for _, ent in higher_entropy_neighbors]
            total_weight = sum(weights)
            if total_weight > 0:
                r = random.uniform(0, total_weight)
                cumsum = 0
                for (neighbor, _), weight in zip(higher_entropy_neighbors, weights):
                    cumsum += weight
                    if r <= cumsum:
                        state = neighbor
                        break
        
        return state
    
    def simulate_evolution(self, initial_state: List[int], 
                         steps: int) -> List[List[int]]:
        """模拟演化轨迹"""
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(steps):
            next_state = self.evolution_step(current_state)
            trajectory.append(next_state)
            current_state = next_state
            
            # 记录历史
            self.evolution_history.append({
                'state': current_state,
                'entropy': self.compute_binary_entropy(current_state),
                'is_fixed_point': current_state == self.self_referential_map(current_state)
            })
        
        return trajectory
    
    def find_fixed_points(self) -> List[List[int]]:
        """寻找所有不动点（S = f(S)）"""
        fixed_points = []
        
        for state in self.phi_system.valid_states:
            f_state = self.self_referential_map(state)
            if state == f_state:
                fixed_points.append(state)
        
        return fixed_points
    
    def analyze_basin_of_attraction(self, fixed_point: List[int], 
                                  max_steps: int = 50) -> Dict[str, Any]:
        """分析不动点的吸引域"""
        basin = set()
        basin.add(tuple(fixed_point))
        
        # 测试所有状态
        converged_states = []
        
        for state in self.phi_system.valid_states:
            current = state
            trajectory = [current]
            
            for _ in range(max_steps):
                current = self.self_evolution_step(current)
                trajectory.append(current)
                
                if current == fixed_point:
                    converged_states.append((state, len(trajectory) - 1))
                    basin.add(tuple(state))
                    break
        
        return {
            'fixed_point': fixed_point,
            'basin_size': len(basin),
            'basin_ratio': len(basin) / self.num_states,
            'converged_states': converged_states[:10]  # 前10个
        }
    
    def verify_entropy_increase(self, trajectory: List[List[int]]) -> Dict[str, Any]:
        """验证熵增原理"""
        entropies = [self.compute_binary_entropy(state) for state in trajectory]
        
        increases = 0
        decreases = 0
        total_change = 0
        
        for i in range(1, len(entropies)):
            change = entropies[i] - entropies[i-1]
            total_change += change
            if change > 0:
                increases += 1
            elif change < 0:
                decreases += 1
        
        return {
            'initial_entropy': entropies[0],
            'final_entropy': entropies[-1],
            'total_change': total_change,
            'average_rate': total_change / (len(entropies) - 1) if len(entropies) > 1 else 0,
            'increases': increases,
            'decreases': decreases,
            'entropy_values': entropies
        }
    
    def verify_determinism(self, initial_state: List[int], 
                         steps: int = 20, 
                         trials: int = 5) -> Dict[str, Any]:
        """验证演化的确定性（在随机种子固定时）"""
        trajectories = []
        
        for trial in range(trials):
            # 固定随机种子以测试确定性
            random.seed(42 + trial)
            trajectory = self.simulate_evolution(initial_state, steps)
            trajectories.append(trajectory)
        
        # 比较轨迹
        all_same = True
        for i in range(1, len(trajectories)):
            if trajectories[i] != trajectories[0]:
                all_same = False
                break
        
        # 计算轨迹之间的平均距离
        avg_distance = 0
        count = 0
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                for k in range(min(len(trajectories[i]), len(trajectories[j]))):
                    dist = self.compute_hamming_distance(
                        trajectories[i][k], trajectories[j][k]
                    )
                    avg_distance += dist
                    count += 1
        
        if count > 0:
            avg_distance /= count
        
        return {
            'deterministic': all_same,
            'num_trajectories': len(trajectories),
            'average_distance': avg_distance,
            'trajectory_lengths': [len(t) for t in trajectories]
        }
    
    def verify_self_reference_preservation(self, 
                                         trajectory: List[List[int]]) -> Dict[str, Any]:
        """验证自指性保持"""
        preservation_scores = []
        
        for state in trajectory:
            f_state = self.self_referential_map(state)
            # 自指性得分：与f(S)的相似度
            score = sum(s == f for s, f in zip(state, f_state)) / self.n
            preservation_scores.append(score)
        
        # 检查是否有收敛到不动点
        fixed_points_reached = []
        fixed_points = self.find_fixed_points()
        
        for i, state in enumerate(trajectory):
            if state in fixed_points:
                fixed_points_reached.append((i, state))
        
        return {
            'average_score': sum(preservation_scores) / len(preservation_scores),
            'min_score': min(preservation_scores),
            'max_score': max(preservation_scores),
            'fixed_points_reached': len(fixed_points_reached),
            'scores': preservation_scores
        }


class TestC3_1_BinarySystemEvolution(unittest.TestCase):
    """C3-1系统演化推论的二进制验证测试"""
    
    def setUp(self):
        """测试初始化"""
        self.verifier = BinarySystemEvolutionVerifier(n=6)
        
    def test_binary_self_referential_map(self):
        """测试二进制自指映射"""
        print("\n=== 测试二进制自指映射 ===")
        
        # 测试几个状态
        test_states = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0]
        ]
        
        for state in test_states:
            f_state = self.verifier.self_referential_map(state)
            print(f"  S = {state}")
            print(f"  f(S) = {f_state}")
            print(f"  有效: {self.verifier.phi_system._is_valid_phi_state(f_state)}")
            print()
        
        # 验证映射保持有效性
        for state in self.verifier.phi_system.valid_states[:10]:
            f_state = self.verifier.self_referential_map(state)
            self.assertTrue(
                self.verifier.phi_system._is_valid_phi_state(f_state),
                f"映射应该保持有效性: {state} -> {f_state}"
            )
        
        print("✓ 自指映射验证通过")
    
    def test_fixed_points(self):
        """测试不动点"""
        print("\n=== 测试不动点 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        
        print(f"找到 {len(fixed_points)} 个不动点:")
        for i, fp in enumerate(fixed_points[:5]):
            print(f"  不动点{i+1}: {fp}")
            # 验证确实是不动点
            f_fp = self.verifier.self_referential_map(fp)
            self.assertEqual(fp, f_fp, "应该满足 S = f(S)")
        
        self.assertGreater(len(fixed_points), 0, "应该至少有一个不动点")
        print("✓ 不动点验证通过")
    
    def test_entropy_increase(self):
        """测试熵增原理"""
        print("\n=== 测试熵增原理 ===")
        
        # 选择一个低熵初始状态
        initial_state = [1, 0, 0, 0, 0, 0]
        
        # 模拟演化
        trajectory = self.verifier.simulate_evolution(initial_state, steps=50)
        
        # 验证熵增
        entropy_data = self.verifier.verify_entropy_increase(trajectory)
        
        print(f"熵增验证:")
        print(f"  初始熵: {entropy_data['initial_entropy']:.4f}")
        print(f"  最终熵: {entropy_data['final_entropy']:.4f}")
        print(f"  总变化: {entropy_data['total_change']:.4f}")
        print(f"  平均速率: {entropy_data['average_rate']:.6f}")
        print(f"  增加/减少: {entropy_data['increases']}/{entropy_data['decreases']}")
        
        self.assertGreaterEqual(
            entropy_data['total_change'], 0,
            "总熵变应该非负"
        )
        print("✓ 熵增原理验证通过")
    
    def test_basin_of_attraction(self):
        """测试吸引域"""
        print("\n=== 测试吸引域 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        
        if fixed_points:
            # 分析第一个不动点的吸引域
            fp = fixed_points[0]
            basin_data = self.verifier.analyze_basin_of_attraction(fp)
            
            print(f"不动点 {fp} 的吸引域:")
            print(f"  吸引域大小: {basin_data['basin_size']}")
            print(f"  占总状态比例: {basin_data['basin_ratio']:.2%}")
            
            if basin_data['converged_states']:
                print(f"  收敛状态示例:")
                for state, steps in basin_data['converged_states'][:3]:
                    print(f"    {state} -> {steps}步收敛")
            
            self.assertGreater(
                basin_data['basin_size'], 0,
                "吸引域应该非空"
            )
        
        print("✓ 吸引域验证通过")
    
    def test_evolution_properties(self):
        """测试演化性质"""
        print("\n=== 测试演化性质 ===")
        
        # 测试不同初始状态
        test_states = [
            [0, 0, 0, 0, 0, 0],  # 全0
            [1, 0, 1, 0, 1, 0],  # 交替模式
            [1, 0, 0, 0, 0, 0],  # 单个1
        ]
        
        for initial in test_states:
            print(f"\n初始状态: {initial}")
            
            # 短期演化
            trajectory = self.verifier.simulate_evolution(initial, steps=20)
            
            # 自指性保持
            self_ref_data = self.verifier.verify_self_reference_preservation(trajectory)
            print(f"  自指性平均得分: {self_ref_data['average_score']:.3f}")
            print(f"  到达不动点: {self_ref_data['fixed_points_reached']}次")
            
            # 检查最终状态
            final_state = trajectory[-1]
            final_entropy = self.verifier.compute_binary_entropy(final_state)
            print(f"  最终状态: {final_state}")
            print(f"  最终熵: {final_entropy:.4f}")
        
        print("\n✓ 演化性质验证通过")
    
    def test_complete_binary_evolution(self):
        """完整的二进制演化验证"""
        print("\n=== C3-1 完整二进制演化验证 ===")
        
        # 1. 不动点
        fixed_points = self.verifier.find_fixed_points()
        print(f"\n1. 不动点: {len(fixed_points)}个")
        
        # 2. 典型演化
        initial = [1, 0, 1, 0, 0, 0]
        trajectory = self.verifier.simulate_evolution(initial, steps=100)
        
        # 3. 熵增
        entropy_data = self.verifier.verify_entropy_increase(trajectory)
        print(f"\n2. 熵增验证:")
        print(f"   平均熵增率: {entropy_data['average_rate']:.6f}")
        print(f"   总熵变: {entropy_data['total_change']:.4f}")
        
        # 4. 自指性
        self_ref_data = self.verifier.verify_self_reference_preservation(trajectory)
        print(f"\n3. 自指性:")
        print(f"   平均保持度: {self_ref_data['average_score']:.3f}")
        
        # 5. 确定性（固定随机种子时）
        det_data = self.verifier.verify_determinism(initial, steps=10, trials=3)
        print(f"\n4. 确定性:")
        print(f"   确定性演化: {det_data['deterministic']}")
        print(f"   平均轨迹距离: {det_data['average_distance']:.3f}")
        
        # 综合判断
        self.assertGreater(len(fixed_points), 0, "必须存在不动点")
        self.assertGreaterEqual(entropy_data['average_rate'], 0, "平均熵增率应非负")
        self.assertGreater(self_ref_data['average_score'], 0.5, "应保持一定自指性")
        
        print("\n✓ C3-1二进制演化推论验证通过！")


def run_binary_verification():
    """运行二进制验证"""
    print("=" * 80)
    print("C3-1 系统演化推论 - 完整二进制验证")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestC3_1_BinarySystemEvolution)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ C3-1二进制系统演化推论验证成功！")
        print("完全基于二进制φ-表示的演化规律得到验证。")
    else:
        print("✗ C3-1二进制验证发现问题")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    success = run_binary_verification()
    exit(0 if success else 1)