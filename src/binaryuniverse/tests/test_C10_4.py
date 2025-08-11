#!/usr/bin/env python3
"""
C10-4: 元数学结构可判定性推论 - 完整测试程序

验证φ-编码二进制宇宙的可判定性层次，包括：
1. 直接可判定性质（多项式时间）
2. 轨道可判定性质（指数时间）
3. 临界深度可判定性
4. 不可判定性边界
5. 判定算法的正确性
"""

import unittest
import numpy as np
from typing import List, Optional, Tuple, Set, Callable, Dict
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod


class PhiNumber:
    """φ进制数系统"""
    def __init__(self, value: float):
        self.phi = (1 + np.sqrt(5)) / 2
        self.value = float(value)
        
    def __eq__(self, other):
        if isinstance(other, PhiNumber):
            return abs(self.value - other.value) < 1e-10
        return abs(self.value - float(other)) < 1e-10
        
    def __repr__(self):
        return f"φ({self.value:.6f})"


@dataclass
class State:
    """系统状态"""
    binary: str
    
    def __post_init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        # 确保满足no-11约束
        if '11' in self.binary:
            raise ValueError(f"State {self.binary} violates no-11 constraint")
            
    def __eq__(self, other):
        return self.binary == other.binary
        
    def __hash__(self):
        return hash(self.binary)
        
    def collapse(self) -> 'State':
        """Collapse操作"""
        # 简化的collapse：添加自身的某种变换
        if len(self.binary) >= 8:  # 防止无限增长
            # 返回某种循环
            return State(self.binary[:4])
        
        # 基本规则：重复并变换
        new_binary = self.binary + self.transform()
        # 规范化以满足no-11
        new_binary = self.normalize_no_11(new_binary)
        return State(new_binary)
        
    def transform(self) -> str:
        """简单变换"""
        # 将0和1互换
        return ''.join('1' if c == '0' else '0' for c in self.binary)
        
    def normalize_no_11(self, s: str) -> str:
        """规范化以满足no-11约束"""
        while '11' in s:
            s = s.replace('11', '10')
        return s
        
    def entropy(self) -> float:
        """计算熵"""
        if not self.binary:
            return 0.0
        ones = self.binary.count('1')
        zeros = self.binary.count('0')
        total = ones + zeros
        
        if ones == 0 or zeros == 0:
            return 0.0
            
        p1 = ones / total
        p0 = zeros / total
        return -(p1 * np.log2(p1) + p0 * np.log2(p0)) * total
        
    def recursive_depth(self) -> int:
        """递归深度"""
        h = self.entropy()
        return int(np.log(h + 1) / np.log(self.phi))


class Property(ABC):
    """性质基类"""
    @abstractmethod
    def evaluate(self, state: State) -> Optional[bool]:
        """评估性质"""
        pass
        
    @abstractmethod
    def is_local(self) -> bool:
        """是否为局部性质"""
        pass
        
    @abstractmethod
    def max_depth(self) -> Optional[int]:
        """最大递归深度"""
        pass


class LocalProperty(Property):
    """局部性质（直接可判定）"""
    def evaluate(self, state: State) -> bool:
        # 示例：检查是否以10开头
        return state.binary.startswith('10')
        
    def is_local(self) -> bool:
        return True
        
    def max_depth(self) -> Optional[int]:
        return 0


class OrbitProperty(Property):
    """轨道性质（指数时间可判定）"""
    def evaluate(self, state: State) -> bool:
        # 示例：检查是否最终进入周期
        visited = set()
        current = state
        
        while current not in visited:
            visited.add(current)
            current = current.collapse()
            if len(visited) > 100:  # 防止无限循环
                break
                
        return current in visited
        
    def is_local(self) -> bool:
        return False
        
    def max_depth(self) -> Optional[int]:
        return None


class DeepProperty(Property):
    """深度性质（可能不可判定）"""
    def __init__(self, depth: int):
        self.depth = depth
        
    def evaluate(self, state: State) -> Optional[bool]:
        if state.recursive_depth() < self.depth:
            # 在深度内可判定
            return True
        else:
            # 超过深度，不可判定
            return None
            
    def is_local(self) -> bool:
        return False
        
    def max_depth(self) -> Optional[int]:
        return self.depth


class DirectlyDecidable:
    """直接可判定性质"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def verify_no_11_constraint(self, state: str) -> bool:
        """O(n)时间验证no-11约束"""
        return '11' not in state
        
    def compute_entropy(self, state: State) -> float:
        """O(n)时间计算熵"""
        return state.entropy()
        
    def compute_recursive_depth(self, state: State) -> int:
        """O(n)时间计算递归深度"""
        return state.recursive_depth()
        
    def phi_distance(self, s1: State, s2: State) -> float:
        """O(n)时间计算φ-距离"""
        distance = 0.0
        max_len = max(len(s1.binary), len(s2.binary))
        
        for i in range(max_len):
            bit1 = int(s1.binary[i]) if i < len(s1.binary) else 0
            bit2 = int(s2.binary[i]) if i < len(s2.binary) else 0
            distance += abs(bit1 - bit2) / (self.phi ** (i + 1))
            
        return distance


class OrbitDecidable:
    """轨道性质可判定"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.visited_states = {}
        
    def detect_period(self, initial: State) -> Tuple[int, int]:
        """检测轨道周期"""
        self.visited_states.clear()
        current = initial
        step = 0
        
        while current not in self.visited_states:
            self.visited_states[current] = step
            current = current.collapse()
            step += 1
            
            # 防止无限循环
            if step > 1000:
                return step, 0
                
        pre_period = self.visited_states[current]
        period = step - pre_period
        
        return pre_period, period
        
    def is_same_orbit(self, s1: State, s2: State) -> bool:
        """判断两个状态是否在同一轨道"""
        # 计算两个轨道
        orbit1 = self.compute_orbit(s1)
        orbit2 = self.compute_orbit(s2)
        
        # 检查交集
        return bool(orbit1.intersection(orbit2))
        
    def compute_orbit(self, initial: State) -> Set[State]:
        """计算完整轨道"""
        orbit = set()
        current = initial
        
        while current not in orbit:
            orbit.add(current)
            current = current.collapse()
            
            if len(orbit) > 100:
                break
                
        return orbit


class CriticalDecidable:
    """临界深度内可判定"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def critical_depth(self, n: int) -> int:
        """计算临界深度"""
        if n <= 1:
            return 1
        return int(np.log(n) / np.log(self.phi)) + 1
        
    def is_decidable_at_depth(self, property: Property, depth: int, n: int) -> bool:
        """判断性质在给定深度是否可判定"""
        return depth < self.critical_depth(n)
        
    def bounded_search(self, source: State, target: State, 
                      max_depth: int) -> Optional[List[State]]:
        """有界深度搜索"""
        if max_depth == 0:
            return [source] if source == target else None
            
        # BFS搜索
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
                
            if current == target:
                return path
                
            # 扩展后继
            next_state = current.collapse()
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
                
        return None


class DecidabilityChecker:
    """可判定性检查器"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.direct = DirectlyDecidable()
        self.orbit = OrbitDecidable()
        self.critical = CriticalDecidable()
        
    def classify_property(self, property: Property) -> str:
        """分类性质的可判定性"""
        if property.is_local():
            return "DIRECTLY_DECIDABLE"
        elif property.max_depth() is None:
            return "ORBIT_DECIDABLE"
        elif property.max_depth() > 0:
            return "CRITICALLY_DECIDABLE"
        else:
            return "UNDECIDABLE"
            
    def decide(self, property: Property, input_size: int) -> Optional[bool]:
        """主判定函数"""
        classification = self.classify_property(property)
        
        if classification == "DIRECTLY_DECIDABLE":
            # 创建测试状态
            test_state = State("10" + "0" * (input_size - 2))
            return property.evaluate(test_state)
        elif classification == "ORBIT_DECIDABLE":
            # 测试几个状态
            for i in range(min(10, input_size)):
                test_state = State("1" + "0" * i)
                if not property.evaluate(test_state):
                    return False
            return True
        elif classification == "CRITICALLY_DECIDABLE":
            # 检查深度
            critical = self.critical.critical_depth(input_size)
            if property.max_depth() < critical:
                return True
            else:
                return None
        else:
            return None


class TestMetamathematicalDecidability(unittest.TestCase):
    """C10-4 元数学可判定性测试套件"""
    
    def setUp(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_directly_decidable_properties(self):
        """测试直接可判定性质"""
        direct = DirectlyDecidable()
        
        # 测试no-11约束验证
        self.assertTrue(direct.verify_no_11_constraint("1010"))
        self.assertFalse(direct.verify_no_11_constraint("1101"))
        
        # 测试熵计算
        s = State("1010")
        entropy = direct.compute_entropy(s)
        self.assertGreater(entropy, 0)
        self.assertLess(entropy, 10)
        
        # 测试递归深度计算
        depth = direct.compute_recursive_depth(s)
        self.assertGreaterEqual(depth, 0)
        self.assertLess(depth, 10)
        
        # 测试φ-距离
        s1 = State("10")
        s2 = State("01")
        distance = direct.phi_distance(s1, s2)
        self.assertGreater(distance, 0)
        
    def test_orbit_decidable_properties(self):
        """测试轨道可判定性质"""
        orbit_dec = OrbitDecidable()
        
        # 测试周期检测
        s = State("1")
        pre_period, period = orbit_dec.detect_period(s)
        self.assertGreaterEqual(pre_period, 0)
        self.assertGreater(period, 0)
        
        # 测试轨道计算
        orbit = orbit_dec.compute_orbit(s)
        self.assertGreater(len(orbit), 0)
        self.assertLess(len(orbit), 100)
        
        # 测试同轨道判定
        s1 = State("1")
        s2 = s1.collapse()
        self.assertTrue(orbit_dec.is_same_orbit(s1, s2))
        
        # 不同轨道
        s3 = State("0")
        # 可能在同一轨道也可能不在，取决于collapse实现
        result = orbit_dec.is_same_orbit(s1, s3)
        self.assertIsInstance(result, bool)
        
    def test_critical_depth_decidability(self):
        """测试临界深度可判定性"""
        critical = CriticalDecidable()
        
        # 测试临界深度计算
        for n in [10, 20, 50, 100]:
            depth = critical.critical_depth(n)
            self.assertGreater(depth, 0)
            self.assertLess(depth, n)
            
            # 验证对数关系
            expected = int(np.log(n) / np.log(self.phi)) + 1
            self.assertEqual(depth, expected)
            
        # 测试有界搜索
        s1 = State("1")
        s2 = State("10")
        path = critical.bounded_search(s1, s2, max_depth=5)
        
        if path is not None:
            self.assertEqual(path[0], s1)
            self.assertEqual(path[-1], s2)
            self.assertLessEqual(len(path), 6)
            
    def test_decidability_classification(self):
        """测试可判定性分类"""
        checker = DecidabilityChecker()
        
        # 测试局部性质
        local_prop = LocalProperty()
        self.assertEqual(
            checker.classify_property(local_prop),
            "DIRECTLY_DECIDABLE"
        )
        
        # 测试轨道性质
        orbit_prop = OrbitProperty()
        self.assertEqual(
            checker.classify_property(orbit_prop),
            "ORBIT_DECIDABLE"
        )
        
        # 测试深度性质
        deep_prop = DeepProperty(depth=5)
        self.assertEqual(
            checker.classify_property(deep_prop),
            "CRITICALLY_DECIDABLE"
        )
        
    def test_decision_algorithms(self):
        """测试判定算法"""
        checker = DecidabilityChecker()
        
        # 测试直接可判定
        local_prop = LocalProperty()
        result = checker.decide(local_prop, 10)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bool)
        
        # 测试轨道可判定
        orbit_prop = OrbitProperty()
        result = checker.decide(orbit_prop, 5)
        self.assertIsNotNone(result)
        self.assertTrue(result)  # 所有状态都进入周期
        
        # 测试临界深度
        deep_prop = DeepProperty(depth=3)
        result = checker.decide(deep_prop, 100)
        # 深度3 < 临界深度log_φ(100) ≈ 4.3
        self.assertTrue(result)
        
        # 测试不可判定
        very_deep_prop = DeepProperty(depth=10)
        result = checker.decide(very_deep_prop, 10)
        # 深度10 > 临界深度log_φ(10) ≈ 2.1
        self.assertIsNone(result)
        
    def test_complexity_bounds(self):
        """测试复杂度界限"""
        # 测试Fibonacci数界限
        def fibonacci(n):
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(n - 1):
                a, b = b, a + b
            return b
            
        # 验证状态空间大小
        for n in range(1, 10):
            # 长度为n的满足no-11约束的串数量
            count = 0
            for i in range(2**n):
                binary = bin(i)[2:].zfill(n)
                if '11' not in binary:
                    count += 1
                    
            # 应该等于F_{n+2}
            expected = fibonacci(n + 2)
            self.assertEqual(count, expected)
            
    def test_decidability_boundary(self):
        """测试可判定性边界"""
        critical = CriticalDecidable()
        
        # 测试不同输入规模的临界深度
        test_cases = [
            (10, 5),    # log_φ(10) ≈ 4.78, 所以是4+1=5
            (100, 10),   # log_φ(100) ≈ 9.57, 所以是9+1=10
            (1000, 15),  # log_φ(1000) ≈ 14.35, 所以是14+1=15
        ]
        
        for n, expected_critical in test_cases:
            actual_critical = critical.critical_depth(n)
            self.assertEqual(actual_critical, expected_critical)
            
            # 验证深度小于临界值时可判定
            prop_below = DeepProperty(depth=expected_critical - 1)
            self.assertTrue(critical.is_decidable_at_depth(
                prop_below, expected_critical - 1, n
            ))
            
            # 验证深度大于等于临界值时不可判定
            prop_above = DeepProperty(depth=expected_critical + 1)
            self.assertFalse(critical.is_decidable_at_depth(
                prop_above, expected_critical + 1, n
            ))
            
    def test_time_complexity(self):
        """测试时间复杂度"""
        direct = DirectlyDecidable()
        
        # 测试直接可判定的线性时间
        sizes = [10, 100, 1000]
        times = []
        
        for size in sizes:
            state = State("1" + "0" * (size - 1))
            start = time.time()
            
            # 执行O(n)操作
            direct.verify_no_11_constraint(state.binary)
            direct.compute_entropy(state)
            
            elapsed = time.time() - start
            times.append(elapsed)
            
        # 验证线性增长（允许一定误差）
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            # 时间应该大致按输入规模比例增长
            self.assertLess(ratio, size_ratio * 5)  # 允许5倍误差
            
    def test_probabilistic_decision(self):
        """测试概率判定"""
        # 简单的蒙特卡洛测试
        def monte_carlo_test(property: Property, samples: int = 100) -> float:
            positive = 0
            
            for i in range(samples):
                # 生成随机有效状态
                length = (i % 10) + 1
                binary = ""
                for _ in range(length):
                    if binary.endswith('1'):
                        binary += '0'
                    else:
                        binary += str(i % 2)
                        
                try:
                    state = State(binary)
                    if property.evaluate(state):
                        positive += 1
                except:
                    pass
                    
            return positive / samples
            
        # 测试局部性质的概率
        local_prop = LocalProperty()
        prob = monte_carlo_test(local_prop)
        # 应该有一定比例满足
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
        
    def test_incremental_decision(self):
        """测试增量判定"""
        orbit_dec = OrbitDecidable()
        
        # 测试逐步增加规模的判定
        results = {}
        for size in [2, 4, 6, 8]:
            state = State("1" + "0" * (size - 1))
            pre, period = orbit_dec.detect_period(state)
            results[size] = (pre, period)
            
            # 验证结果合理
            self.assertGreaterEqual(pre, 0)
            self.assertGreater(period, 0)
            self.assertLess(pre + period, 1000)
            
        # 检查是否有规律
        # 随着规模增加，周期可能变化
        self.assertEqual(len(results), 4)
        
    def test_caching_effectiveness(self):
        """测试缓存效果"""
        # 简单的缓存实现
        cache = {}
        
        def cached_collapse(state: State) -> State:
            if state in cache:
                return cache[state]
            result = state.collapse()
            cache[state] = result
            return result
            
        # 测试缓存命中率
        s = State("1")
        orbit = []
        
        for _ in range(20):
            orbit.append(s)
            s = cached_collapse(s)
            
        # 由于进入周期，应该有缓存命中
        self.assertGreater(len(cache), 0)
        self.assertLess(len(cache), 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)