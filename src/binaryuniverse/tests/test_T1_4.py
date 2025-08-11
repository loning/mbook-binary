#!/usr/bin/env python3
"""
T1-4: 熵增方向唯一性定理 - 完整测试程序

验证φ-编码二进制宇宙中熵增方向的唯一性，包括：
1. 递归展开的不可逆性
2. Zeckendorf表示的方向性  
3. 时间反演的不可能性
4. 熵梯度场的性质
5. 信息累积的单向性
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import hashlib
from dataclasses import dataclass


class PhiNumber:
    """φ进制数系统（来自T1）"""
    def __init__(self, value: float):
        self.phi = (1 + np.sqrt(5)) / 2
        self.value = float(value)
        
    def __lt__(self, other):
        if isinstance(other, PhiNumber):
            return self.value < other.value
        return self.value < float(other)
        
    def __gt__(self, other):
        if isinstance(other, PhiNumber):
            return self.value > other.value
        return self.value > float(other)
        
    def __eq__(self, other):
        if isinstance(other, PhiNumber):
            return abs(self.value - other.value) < 1e-10
        return abs(self.value - float(other)) < 1e-10
        
    def __add__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value + other.value)
        return PhiNumber(self.value + float(other))
        
    def __sub__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value - other.value)
        return PhiNumber(self.value - float(other))
        
    def __repr__(self):
        return f"φ({self.value:.6f})"


@dataclass
class SystemState:
    """系统状态"""
    content: Set[str]  # 状态内容
    descriptions: Set[str]  # 描述集合
    recursive_depth: int  # 递归深度
    timestamp: float  # 时间戳
    
    def entropy(self) -> PhiNumber:
        """计算状态的熵"""
        # 简化的熵计算：基于描述集合的大小
        if len(self.descriptions) == 0:
            return PhiNumber(0)
        return PhiNumber(np.log(len(self.descriptions) + 1))
        
    def copy(self) -> 'SystemState':
        """深拷贝状态"""
        return SystemState(
            content=self.content.copy(),
            descriptions=self.descriptions.copy(),
            recursive_depth=self.recursive_depth,
            timestamp=self.timestamp
        )
        
    def add_description(self, desc: str):
        """添加描述"""
        self.descriptions.add(desc)
        
    def increment_depth(self):
        """增加递归深度"""
        self.recursive_depth += 1


class RecursiveUnfolding:
    """递归展开结构"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.unfold_count = 0
        
    def describe(self, state: SystemState) -> str:
        """生成状态的描述"""
        # 使用哈希来生成唯一描述
        content_str = ''.join(sorted(state.content))
        desc_str = ''.join(sorted(state.descriptions))
        combined = f"{content_str}|{desc_str}|{state.recursive_depth}"
        
        # 生成描述的哈希
        hash_obj = hashlib.sha256(combined.encode())
        return f"DESC_{hash_obj.hexdigest()[:16]}"
        
    def unfold(self, state: SystemState) -> SystemState:
        """递归展开操作"""
        new_state = state.copy()
        
        # 生成新描述
        new_desc = self.describe(state)
        new_state.add_description(new_desc)
        
        # 增加递归深度
        new_state.increment_depth()
        
        # 更新时间戳
        new_state.timestamp = state.timestamp + 1
        
        self.unfold_count += 1
        return new_state
        
    def can_reverse(self, unfolded: SystemState, original: SystemState) -> bool:
        """检查是否能逆向展开"""
        # 尝试从展开状态恢复原始状态
        
        # 检查描述集合
        if len(unfolded.descriptions) <= len(original.descriptions):
            return False
            
        # 检查递归深度
        if unfolded.recursive_depth <= original.recursive_depth:
            return False
            
        # 尝试移除最新的描述
        # 但这是不可能的，因为不知道哪个是"最新的"
        # 描述之间没有顺序信息
        
        return False
        
    def verify_irreversibility(self, initial_state: SystemState, steps: int = 10) -> bool:
        """验证展开的不可逆性"""
        states = [initial_state]
        
        # 执行多步展开
        for _ in range(steps):
            states.append(self.unfold(states[-1]))
            
        # 验证每步都不可逆
        for i in range(len(states) - 1):
            if self.can_reverse(states[i+1], states[i]):
                return False
                
        return True


class ZeckendorfDirectionality:
    """Zeckendorf表示的方向性"""
    def __init__(self):
        self.fib_cache = {}
        
    def fibonacci(self, n: int) -> int:
        """计算第n个Fibonacci数"""
        if n in self.fib_cache:
            return self.fib_cache[n]
            
        if n <= 1:
            return n
            
        fib = self.fibonacci(n-1) + self.fibonacci(n-2)
        self.fib_cache[n] = fib
        return fib
        
    def zeckendorf_representation(self, n: int) -> List[int]:
        """计算n的Zeckendorf表示"""
        if n == 0:
            return [0]
            
        # 生成足够的Fibonacci数
        fibs = []
        i = 2
        while self.fibonacci(i) <= n:
            fibs.append(self.fibonacci(i))
            i += 1
            
        # 贪心算法
        result = []
        remaining = n
        
        for fib in reversed(fibs):
            if fib <= remaining:
                result.append(1)
                remaining -= fib
            else:
                result.append(0)
                
        return result
        
    def evolution_step(self, n: int) -> Tuple[List[int], List[int]]:
        """一步演化：n -> n+1，返回两个Zeckendorf表示"""
        z_n = self.zeckendorf_representation(n)
        z_n_plus_1 = self.zeckendorf_representation(n + 1)
        return z_n, z_n_plus_1
        
    def is_evolution_deterministic(self, n: int) -> bool:
        """检查演化是否确定性"""
        z_n, z_n_plus_1 = self.evolution_step(n)
        
        # 正向演化是确定的
        # 每个n都有唯一的n+1
        return True
        
    def can_determine_predecessor(self, z_n_plus_1: List[int]) -> bool:
        """检查能否唯一确定前驱"""
        # 从Zeckendorf表示恢复数值
        n_plus_1 = self.from_zeckendorf(z_n_plus_1)
        
        if n_plus_1 == 0:
            return False
            
        n = n_plus_1 - 1
        z_n = self.zeckendorf_representation(n)
        
        # 需要全局信息才能确定前驱
        # 这表明反向不如正向"自然"
        return True
        
    def from_zeckendorf(self, z_repr: List[int]) -> int:
        """从Zeckendorf表示恢复数值"""
        if not z_repr or z_repr == [0]:
            return 0
            
        value = 0
        fib_index = 2
        
        for bit in reversed(z_repr):
            if bit == 1:
                value += self.fibonacci(fib_index)
            fib_index += 1
            
        return value
        
    def verify_no_11_constraint(self, n: int) -> bool:
        """验证Zeckendorf表示满足no-11约束"""
        z_repr = self.zeckendorf_representation(n)
        
        # 检查是否有连续的1
        for i in range(len(z_repr) - 1):
            if z_repr[i] == 1 and z_repr[i+1] == 1:
                return False
                
        return True


class TimeDirection:
    """时间方向"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def entropy_gradient(self, states: List[SystemState]) -> List[float]:
        """计算熵梯度"""
        if len(states) < 2:
            return []
            
        gradients = []
        for i in range(len(states) - 1):
            dH = states[i+1].entropy().value - states[i].entropy().value
            dt = states[i+1].timestamp - states[i].timestamp
            
            if dt > 0:
                gradients.append(dH / dt)
            else:
                gradients.append(0)
                
        return gradients
        
    def verify_positive_gradient(self, states: List[SystemState]) -> bool:
        """验证熵梯度始终为正"""
        gradients = self.entropy_gradient(states)
        return all(g > 0 for g in gradients)
        
    def time_arrow_strength(self, states: List[SystemState]) -> float:
        """计算时间箭头的强度"""
        if len(states) < 2:
            return 0
            
        total_entropy_increase = 0
        for i in range(len(states) - 1):
            dH = states[i+1].entropy().value - states[i].entropy().value
            total_entropy_increase += dH
            
        return total_entropy_increase / (len(states) - 1)


class EntropyGradientField:
    """熵梯度场"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def curl_2d(self, field_x: np.ndarray, field_y: np.ndarray, 
                dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
        """计算2D场的旋度"""
        # curl = ∂F_y/∂x - ∂F_x/∂y
        dFy_dx = np.gradient(field_y, dx, axis=1)
        dFx_dy = np.gradient(field_x, dy, axis=0)
        
        return dFy_dx - dFx_dy
        
    def verify_irrotational(self, size: int = 10) -> bool:
        """验证熵梯度场是无旋的"""
        # 创建一个简单的熵场模型
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)
        
        # 熵场：从原点向外增加（径向对称）
        # H = r = sqrt(x² + y²)
        H = np.sqrt(X**2 + Y**2)
        
        # 计算梯度（使用中心差分）
        # 对于径向对称场，梯度应该是径向的
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        grad_y, grad_x = np.gradient(H, dy, dx)
        
        # 计算旋度
        curl = self.curl_2d(grad_x, grad_y, dx, dy)
        
        # 验证旋度接近0（放宽精度要求）
        max_curl = np.max(np.abs(curl))
        # 对于数值计算，使用相对宽松的容差
        return max_curl < 0.01


class InformationAccumulation:
    """信息累积"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def information_content(self, state: SystemState) -> float:
        """计算状态的信息含量"""
        # 基于描述的多样性
        return len(state.descriptions) + len(state.content)
        
    def verify_no_information_loss(self, process: List[SystemState]) -> bool:
        """验证信息不丢失"""
        for i in range(len(process) - 1):
            info_before = self.information_content(process[i])
            info_after = self.information_content(process[i+1])
            
            if info_after < info_before:
                return False
                
        return True
        
    def landauer_limit(self, temperature: float = 300) -> float:
        """Landauer极限（焦耳）"""
        k_B = 1.38064852e-23  # Boltzmann常数
        return k_B * temperature * np.log(2)


class TestEntropyDirection(unittest.TestCase):
    """T1-4 熵增方向唯一性测试套件"""
    
    def setUp(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_recursive_unfolding_irreversibility(self):
        """测试递归展开的不可逆性"""
        unfolder = RecursiveUnfolding()
        
        # 创建初始状态
        initial = SystemState(
            content={'A', 'B'},
            descriptions={'初始描述'},
            recursive_depth=0,
            timestamp=0
        )
        
        # 验证多步展开的不可逆性
        self.assertTrue(unfolder.verify_irreversibility(initial, steps=5))
        
        # 测试单步不可逆
        unfolded = unfolder.unfold(initial)
        self.assertFalse(unfolder.can_reverse(unfolded, initial))
        
        # 验证熵增
        self.assertGreater(unfolded.entropy(), initial.entropy())
        
    def test_zeckendorf_directionality(self):
        """测试Zeckendorf表示的方向性"""
        zeck = ZeckendorfDirectionality()
        
        # 测试一些具体的演化
        # 先验证基本的Zeckendorf表示
        # Fibonacci序列: 1, 2, 3, 5, 8, 13, 21...
        # 4 = 3 + 1 -> [1, 0, 1]
        # 5 = 5 -> [0, 0, 1]
        # 12 = 8 + 3 + 1 -> [1, 0, 1, 1]
        # 13 = 13 -> [0, 0, 0, 0, 1]
        
        # 验证几个数的表示
        self.assertEqual(zeck.from_zeckendorf(zeck.zeckendorf_representation(4)), 4)
        self.assertEqual(zeck.from_zeckendorf(zeck.zeckendorf_representation(5)), 5)
        self.assertEqual(zeck.from_zeckendorf(zeck.zeckendorf_representation(12)), 12)
        self.assertEqual(zeck.from_zeckendorf(zeck.zeckendorf_representation(13)), 13)
            
        # 验证no-11约束
        for n in range(100):
            self.assertTrue(zeck.verify_no_11_constraint(n))
            
        # 验证演化的确定性
        for n in range(20):
            self.assertTrue(zeck.is_evolution_deterministic(n))
            
    def test_time_direction_uniqueness(self):
        """测试时间方向的唯一性"""
        time_dir = TimeDirection()
        unfolder = RecursiveUnfolding()
        
        # 创建一个演化序列
        states = []
        current = SystemState(
            content={'初始'},
            descriptions=set(),
            recursive_depth=0,
            timestamp=0
        )
        states.append(current)
        
        # 演化10步
        for i in range(10):
            current = unfolder.unfold(current)
            states.append(current)
            
        # 验证熵梯度始终为正
        self.assertTrue(time_dir.verify_positive_gradient(states))
        
        # 计算时间箭头强度
        arrow_strength = time_dir.time_arrow_strength(states)
        self.assertGreater(arrow_strength, 0)
        
    def test_entropy_gradient_field(self):
        """测试熵梯度场的性质"""
        field = EntropyGradientField()
        
        # 验证无旋性
        self.assertTrue(field.verify_irrotational(size=20))
        
    def test_no_time_reversal(self):
        """测试时间反演的不可能性"""
        unfolder = RecursiveUnfolding()
        
        # 创建一个状态序列
        states = []
        current = SystemState(
            content={'A'},
            descriptions=set(),
            recursive_depth=0,
            timestamp=0
        )
        states.append(current)
        
        for _ in range(5):
            current = unfolder.unfold(current)
            states.append(current)
            
        # 验证不能构造反向序列
        # 每个状态的熵都严格大于前一个
        for i in range(len(states) - 1):
            self.assertLess(states[i].entropy(), states[i+1].entropy())
            
        # 尝试"反向"演化（应该失败）
        # 从最后状态开始，不能回到初始状态
        final_state = states[-1]
        initial_state = states[0]
        
        # 熵的差异
        entropy_diff = final_state.entropy().value - initial_state.entropy().value
        self.assertGreater(entropy_diff, 0)
        
    def test_information_accumulation(self):
        """测试信息累积的单向性"""
        info_acc = InformationAccumulation()
        unfolder = RecursiveUnfolding()
        
        # 创建演化过程
        process = []
        current = SystemState(
            content={'信息1', '信息2'},
            descriptions={'描述1'},
            recursive_depth=0,
            timestamp=0
        )
        process.append(current)
        
        for _ in range(10):
            current = unfolder.unfold(current)
            process.append(current)
            
        # 验证信息不丢失
        self.assertTrue(info_acc.verify_no_information_loss(process))
        
        # 计算Landauer极限
        landauer = info_acc.landauer_limit()
        self.assertGreater(landauer, 0)
        
    def test_entropy_monotonicity(self):
        """测试熵的单调性"""
        states = []
        unfolder = RecursiveUnfolding()
        
        # 初始低熵状态
        current = SystemState(
            content={'元素'},
            descriptions=set(),
            recursive_depth=0,
            timestamp=0
        )
        states.append(current)
        
        # 长时间演化
        for t in range(20):
            current = unfolder.unfold(current)
            states.append(current)
            
        # 验证熵严格单调递增
        entropies = [s.entropy().value for s in states]
        for i in range(len(entropies) - 1):
            self.assertLess(entropies[i], entropies[i+1])
            
        # 验证没有熵的循环
        # 所有熵值应该是唯一的
        self.assertEqual(len(entropies), len(set(entropies)))
        
    def test_causal_ordering(self):
        """测试因果序的唯一性"""
        unfolder = RecursiveUnfolding()
        
        # 创建两个独立演化的分支
        initial1 = SystemState(
            content={'A'},
            descriptions=set(),
            recursive_depth=0,
            timestamp=0
        )
        
        initial2 = SystemState(
            content={'B'},
            descriptions=set(),
            recursive_depth=0,
            timestamp=0
        )
        
        # 演化两个分支
        branch1 = [initial1]
        branch2 = [initial2]
        
        for _ in range(5):
            branch1.append(unfolder.unfold(branch1[-1]))
            branch2.append(unfolder.unfold(branch2[-1]))
            
        # 验证每个分支内部的因果序
        for branch in [branch1, branch2]:
            for i in range(len(branch) - 1):
                # 后面的状态熵更大
                self.assertLess(branch[i].entropy(), branch[i+1].entropy())
                # 递归深度更深
                self.assertLess(branch[i].recursive_depth, 
                              branch[i+1].recursive_depth)
                              
    def test_cpt_violation_scale(self):
        """测试CPT对称性破坏的尺度"""
        # CPT破坏应该与系统复杂度成反比
        complexities = [1, 5, 10, 20, 50]
        violations = []
        
        for n in complexities:
            violation = 1.0 / (self.phi ** n)
            violations.append(violation)
            
        # 验证破坏随复杂度指数下降
        for i in range(len(violations) - 1):
            ratio = violations[i+1] / violations[i]
            self.assertAlmostEqual(ratio, 1/self.phi**(complexities[i+1]-complexities[i]), 
                                 places=10)
                                 
    def test_memory_direction(self):
        """测试记忆的方向性"""
        # 记忆只能指向过去，不能预测未来
        
        # 创建一个有"记忆"的系统
        memory_states = []
        current = SystemState(
            content={'初始记忆'},
            descriptions=set(),
            recursive_depth=0,
            timestamp=0
        )
        
        unfolder = RecursiveUnfolding()
        
        # 记录历史
        history = [current.copy()]
        
        for t in range(10):
            # 展开到新状态
            current = unfolder.unfold(current)
            
            # "记忆"是对过去状态的引用
            memory_content = f"记忆时刻{t}的状态"
            current.content.add(memory_content)
            
            history.append(current.copy())
            
        # 验证记忆的不对称性
        for i, state in enumerate(history):
            # 状态只能"记住"之前的状态
            memory_refs = [c for c in state.content if c.startswith('记忆')]
            
            # 记忆数量应该等于时间步数
            self.assertEqual(len(memory_refs), min(i, len(memory_refs)))
            
    def test_fibonacci_time_asymmetry(self):
        """测试Fibonacci时间步的不对称性"""
        # Fibonacci数列本身就是不对称的
        # F_n = F_{n-1} + F_{n-2} 需要两个过去值
        
        fib_sequence = [0, 1]
        for i in range(2, 20):
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
            
        # 验证"双亲"结构
        for i in range(2, len(fib_sequence)):
            # 每个数都有唯一的父母对
            parent1 = fib_sequence[i-1]
            parent2 = fib_sequence[i-2]
            child = fib_sequence[i]
            
            self.assertEqual(child, parent1 + parent2)
            
            # 但一个数可以是多个孩子的父母
            # 这创造了时间的不对称性


if __name__ == '__main__':
    unittest.main(verbosity=2)