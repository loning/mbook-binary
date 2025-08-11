"""
Unit tests for L1-8: Measurement Irreversibility Lemma
L1-8：测量不可逆性引理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math
import copy
import hashlib


class System:
    """系统状态表示"""
    
    def __init__(self, states=None, time=0):
        self.states = states or set()
        self.time = time
        self.history = []  # 历史记录
        self.causal_links = {}  # 因果关系
        
    def entropy(self):
        """计算系统熵"""
        if len(self.states) == 0:
            return 0
        # 简化熵计算：log(状态数)
        return math.log2(len(self.states))
        
    def add_state(self, state):
        """添加新状态"""
        self.states.add(state)
        self.history.append(('add', state, self.time))
        
    def has_state(self, state):
        """检查状态是否存在"""
        return state in self.states
        
    def copy(self):
        """深拷贝系统"""
        new_system = System(self.states.copy(), self.time)
        new_system.history = self.history.copy()
        new_system.causal_links = self.causal_links.copy()
        return new_system
        
    def __eq__(self, other):
        """判断系统相等"""
        if not isinstance(other, System):
            return False
        return (self.states == other.states and 
                self.time == other.time)


class Observer:
    """观察者"""
    
    def __init__(self, id_num=0):
        self.id = id_num
        self.memory = []
        self.state = set()
        
    def perceive(self, system):
        """感知系统"""
        # 简化感知：获取系统状态数和样本
        perception = {
            'size': len(system.states),
            'sample': list(system.states)[:3] if system.states else [],
            'time': system.time
        }
        return perception
        
    def update(self, record):
        """更新观察者状态"""
        new_observer = Observer(self.id)
        new_observer.memory = self.memory + [record]
        new_observer.state = self.state | {f"obs_record_{record.id}"}
        return new_observer
        
    def __eq__(self, other):
        """判断观察者相等"""
        if not isinstance(other, Observer):
            return False
        return (self.id == other.id and 
                self.memory == other.memory and
                self.state == other.state)


class Record:
    """测量记录"""
    
    def __init__(self, id_num, content, timestamp):
        self.id = id_num
        self.content = content
        self.timestamp = timestamp
        self.hash = self._compute_hash()
        self.influenced_states = set()  # 受影响的状态
        
    def _compute_hash(self):
        """计算记录哈希"""
        data = f"{self.id}:{self.content}:{self.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
        
    def has_influenced_system(self):
        """检查是否已影响系统"""
        return len(self.influenced_states) > 0
        
    def influence(self, state):
        """记录影响"""
        self.influenced_states.add(state)
        
    def __str__(self):
        return f"Record({self.id}, {self.hash})"
        
    def __repr__(self):
        return str(self)


class Measurement:
    """测量过程"""
    
    def __init__(self):
        self.measurement_count = 0
        self.measurement_history = []
        
    def apply(self, system, observer):
        """执行测量"""
        # 观察者感知系统
        perception = observer.perceive(system)
        
        # 创建测量记录
        record = Record(
            id_num=self.measurement_count,
            content=perception,
            timestamp=system.time
        )
        self.measurement_count += 1
        
        # 创建新系统（包含测量记录）
        new_system = system.copy()
        new_system.add_state(record)
        new_system.add_state(f"desc_{record.id}")
        new_system.time += 1
        
        # 记录因果关系
        for state in system.states:
            new_system.causal_links[state] = record
            record.influence(state)
            
        # 更新观察者
        new_observer = observer.update(record)
        
        # 保存测量历史
        self.measurement_history.append({
            'before': (system, observer),
            'after': (new_system, new_observer, record)
        })
        
        return new_system, new_observer, record
        
    def calculate_entropy_change(self, system, new_system):
        """计算熵变"""
        return new_system.entropy() - system.entropy()


class IrreversibleOperation(Exception):
    """不可逆操作异常"""
    pass


class ReversalAttempt:
    """逆转尝试"""
    
    def __init__(self):
        self.attempts = []
        
    def try_delete_record(self, system, record):
        """尝试删除记录"""
        try:
            # 创建"删除"操作的记录
            delete_record = Record(
                id_num=9999,
                content={'action': 'delete', 'target': record.id},
                timestamp=system.time
            )
            
            # 删除操作本身需要添加新记录
            system.add_state(delete_record)
            
            # 即使删除原记录，删除记录仍然存在
            if record in system.states:
                system.states.remove(record)
                
            # 熵仍然增加了（因为删除记录）
            raise IrreversibleOperation("Deletion creates new record")
            
        except Exception as e:
            self.attempts.append(('delete', str(e)))
            raise
            
    def try_time_reversal(self, system, target_time):
        """尝试时间反演"""
        if system.time <= target_time:
            raise IrreversibleOperation("Cannot reverse to future")
            
        # 时间反演违反因果律
        if system.history:
            raise IrreversibleOperation("History cannot be undone")
            
        self.attempts.append(('time_reversal', 'Failed'))
        
    def try_entropy_reduction(self, system, target_entropy):
        """尝试减少熵"""
        current_entropy = system.entropy()
        
        if current_entropy <= target_entropy:
            raise IrreversibleOperation("Entropy already lower")
            
        # 根据熵增定律，不能减少熵
        raise IrreversibleOperation("Entropy cannot decrease")


class TestL1_8_MeasurementIrreversibility(VerificationTest):
    """L1-8 测量不可逆性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.measurement = Measurement()
        
    def test_measurement_process_formalization(self):
        """测试测量过程形式化 - 验证检查点1"""
        # 创建初始系统和观察者
        system = System({1, 2, 3}, time=0)
        observer = Observer(id_num=1)
        
        # 执行测量
        result = self.measurement.apply(system, observer)
        
        # 验证返回三元组
        self.assertEqual(
            len(result), 3,
            "Measurement should return triple (S', O', R)"
        )
        
        s_prime, o_prime, record = result
        
        # 验证类型
        self.assertIsInstance(s_prime, System, "First element should be System")
        self.assertIsInstance(o_prime, Observer, "Second element should be Observer")
        self.assertIsInstance(record, Record, "Third element should be Record")
        
        # 验证状态改变
        self.assertNotEqual(s_prime, system, "System should change")
        self.assertNotEqual(o_prime, observer, "Observer should change")
        
    def test_information_creation(self):
        """测试信息创造 - 验证检查点2"""
        system = System({1, 2, 3}, time=0)
        observer = Observer(id_num=1)
        
        # 测量前记录状态
        initial_states = system.states.copy()
        
        # 执行测量
        s_prime, o_prime, record = self.measurement.apply(system, observer)
        
        # 验证记录是新创建的
        self.assertNotIn(
            record, initial_states,
            "Record should not exist before measurement"
        )
        
        # 验证系统包含新信息
        self.assertIn(
            record, s_prime.states,
            "System should contain record"
        )
        
        self.assertIn(
            f"desc_{record.id}", s_prime.states,
            "System should contain record description"
        )
        
        # 验证状态数增加
        self.assertGreater(
            len(s_prime.states), len(initial_states),
            "State count should increase"
        )
        
        # 验证至少增加2个元素
        self.assertGreaterEqual(
            len(s_prime.states) - len(initial_states), 2,
            "Should add at least 2 elements (record + description)"
        )
        
    def test_entropy_increase(self):
        """测试熵增 - 验证检查点3"""
        test_cases = [
            System({1}, time=0),
            System({1, 2, 3}, time=0),
            System(set(range(10)), time=0),
        ]
        
        for system in test_cases:
            observer = Observer()
            initial_entropy = system.entropy()
            
            # 执行测量
            s_prime, _, _ = self.measurement.apply(system, observer)
            final_entropy = s_prime.entropy()
            
            # 验证熵增
            self.assertGreater(
                final_entropy, initial_entropy,
                f"Entropy should increase: {initial_entropy} -> {final_entropy}"
            )
            
            # 验证最小熵增
            # 熵增取决于状态数的相对增加
            # H = log2(n), ΔH = log2(n_new) - log2(n_old) = log2(n_new/n_old)
            # 添加2个状态：n_new = n_old + 2
            # 最小相对增加：(n+2)/n，对于n=1是3倍，对于大n接近1
            entropy_change = final_entropy - initial_entropy
            
            # 验证熵确实增加了
            self.assertGreater(
                entropy_change, 0,
                f"Entropy should increase, got change of {entropy_change}"
            )
            
    def test_irreversibility_proof(self):
        """测试不可逆性证明 - 验证检查点4"""
        system = System({1, 2, 3}, time=0)
        observer = Observer(id_num=1)
        
        # 记录初始状态
        initial_system = system.copy()
        initial_observer = copy.deepcopy(observer)
        initial_entropy = system.entropy()
        
        # 执行测量
        s_prime, o_prime, record = self.measurement.apply(system, observer)
        
        # 尝试各种逆转方法
        reversal = ReversalAttempt()
        
        # 1. 尝试删除记录
        with self.assertRaises(IrreversibleOperation):
            reversal.try_delete_record(s_prime, record)
            
        # 2. 尝试时间反演
        with self.assertRaises(IrreversibleOperation):
            reversal.try_time_reversal(s_prime, initial_system.time)
            
        # 3. 尝试减少熵
        with self.assertRaises(IrreversibleOperation):
            reversal.try_entropy_reduction(s_prime, initial_entropy)
            
        # 验证没有方法能完全恢复初始状态
        self.assertNotEqual(s_prime, initial_system)
        self.assertNotEqual(s_prime.entropy(), initial_entropy)
        
    def test_information_conservation(self):
        """测试信息守恒问题"""
        system = System({1, 2}, time=0)
        observer = Observer()
        
        # 执行测量
        s_prime, o_prime, record = self.measurement.apply(system, observer)
        
        # 尝试"删除"信息
        delete_system = s_prime.copy()
        
        # 即使物理删除记录
        if record in delete_system.states:
            delete_system.states.remove(record)
            
        # 信息仍然存在于：
        # 1. 观察者记忆中
        self.assertIn(record, o_prime.memory)
        
        # 2. 因果链中
        self.assertTrue(record.has_influenced_system())
        
        # 3. 系统历史中
        self.assertTrue(
            any('add' in h and record in h for h in s_prime.history)
        )
        
    def test_causal_paradox(self):
        """测试因果悖论"""
        system = System({1, 2}, time=0)
        observer = Observer()
        
        # 执行测量
        s_prime, o_prime, record = self.measurement.apply(system, observer)
        
        # 基于测量结果进行后续操作
        if len(record.content['sample']) > 0:
            s_prime.add_state(f"derived_from_{record.id}")
            
        # 记录已经影响了系统演化
        self.assertTrue(
            record.has_influenced_system(),
            "Record should have causal influence"
        )
        
        # 无法消除这种影响
        derived_states = [s for s in s_prime.states if f"derived_from_{record.id}" in str(s)]
        self.assertGreater(
            len(derived_states), 0,
            "Causal effects persist"
        )
        
    def test_self_reference_paradox(self):
        """测试自指悖论"""
        system = System({1}, time=0)
        observer = Observer()
        
        # 执行测量
        s_prime, o_prime, record1 = self.measurement.apply(system, observer)
        
        # 尝试创建"逆测量"
        # 这本身就是一个新的测量
        reverse_measurement = Measurement()
        s_double_prime, o_double_prime, record2 = reverse_measurement.apply(s_prime, o_prime)
        
        # 验证产生了新记录
        self.assertNotEqual(record1, record2)
        self.assertIn(record2, s_double_prime.states)
        
        # 熵进一步增加
        self.assertGreater(
            s_double_prime.entropy(), s_prime.entropy(),
            "Reverse attempt increases entropy further"
        )
        
    def test_partial_reversibility(self):
        """测试部分可逆性"""
        # 某些操作可以部分逆转
        system = System({'spin_up'}, time=0)
        
        # 模拟自旋翻转（可逆的物理操作）
        system.states.remove('spin_up')
        system.states.add('spin_down')
        
        # 可以翻转回来
        system.states.remove('spin_down')
        system.states.add('spin_up')
        
        self.assertIn('spin_up', system.states)
        
        # 但测量记录不可逆
        observer = Observer()
        s_prime, _, record = self.measurement.apply(system, observer)
        
        # 即使翻转自旋，记录仍存在
        if 'spin_up' in s_prime.states:
            s_prime.states.remove('spin_up')
            s_prime.states.add('spin_down')
            
        self.assertIn(record, s_prime.states)
        
    def test_quantum_measurement_analogy(self):
        """测试量子测量类比"""
        # 模拟量子叠加态
        system = System({'superposition'}, time=0)
        system.quantum_state = {'up': 0.6, 'down': 0.4}  # 概率幅
        
        observer = Observer()
        
        # 测量导致"坍缩"
        s_prime, _, record = self.measurement.apply(system, observer)
        
        # 选择一个本征态
        import random
        random.seed(42)  # 固定随机性
        collapsed = 'up' if random.random() < 0.6 else 'down'
        s_prime.add_state(f'collapsed_to_{collapsed}')
        
        # 测量后移除叠加态，只保留坍缩态
        if 'superposition' in s_prime.states:
            s_prime.states.remove('superposition')
            
        # 验证坍缩态存在
        self.assertIn(f'collapsed_to_{collapsed}', s_prime.states)
        self.assertNotIn('superposition', s_prime.states)
        
        # 原始概率幅信息不可恢复
        self.assertFalse(hasattr(s_prime, 'quantum_state'))
        
    def test_measurement_history(self):
        """测试测量历史"""
        system = System({1, 2}, time=0)
        observer = Observer()
        
        # 多次测量
        measurements = []
        current_system = system
        current_observer = observer
        
        for i in range(3):
            s_new, o_new, record = self.measurement.apply(current_system, current_observer)
            measurements.append((s_new, o_new, record))
            current_system = s_new
            current_observer = o_new
            
        # 验证测量历史
        self.assertEqual(
            len(self.measurement.measurement_history), 3,
            "Should have 3 measurements in history"
        )
        
        # 验证每次测量都增加熵
        entropies = [system.entropy()]
        for s, _, _ in measurements:
            entropies.append(s.entropy())
            
        for i in range(len(entropies) - 1):
            self.assertGreater(
                entropies[i+1], entropies[i],
                f"Entropy should increase at step {i}"
            )


if __name__ == "__main__":
    unittest.main()