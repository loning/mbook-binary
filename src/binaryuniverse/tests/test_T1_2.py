"""
Unit tests for T1-2: Five-Fold Equivalence Theorem
T1-2：五重等价性定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class StateTransition:
    """状态转换记录"""
    
    def __init__(self, from_state, to_state, timestamp):
        self.from_state = from_state
        self.to_state = to_state
        self.timestamp = timestamp
        
    def __repr__(self):
        return f"Transition({self.from_state}->{self.to_state}@{self.timestamp})"
        
    def __hash__(self):
        return hash((self.from_state, self.to_state, self.timestamp))
        
    def __eq__(self, other):
        return (isinstance(other, StateTransition) and 
                self.from_state == other.from_state and
                self.to_state == other.to_state and
                self.timestamp == other.timestamp)


class InformationRecord:
    """信息记录"""
    
    def __init__(self, content, source, timestamp):
        self.content = content
        self.source = source
        self.timestamp = timestamp
        
    def __repr__(self):
        return f"Info({self.content}@{self.timestamp})"
        
    def __hash__(self):
        return hash((str(self.content), str(self.source), self.timestamp))


class Observer:
    """系统观察者"""
    
    def __init__(self, system, obs_id):
        self.system = system
        self.id = obs_id
        self.observations = []
        
    def observe(self, target):
        """观察目标并创建记录"""
        # 创建观察记录
        record = f"obs_{self.id}_of_{target}_at_{self.system.current_time}"
        
        # 记录观察
        self.observations.append({
            'target': target,
            'record': record,
            'time': self.system.current_time
        })
        
        return record
        
    def process_information(self, info):
        """处理信息"""
        # 简单的信息处理：返回信息数量
        if isinstance(info, (list, set)):
            return len(info)
        return 1
        
    def __repr__(self):
        return f"Observer({self.id})"
        
    def __hash__(self):
        return hash(self.id)


class FiveFoldSystem:
    """五重等价性验证系统"""
    
    def __init__(self):
        self.states = [set(['initial'])]  # 状态序列
        self.descriptions = [{'initial': 'init_desc'}]  # 描述序列
        self.transitions = []  # 转换记录
        self.information = [set()]  # 信息集合序列
        self.observers = []  # 观察者列表
        self.current_time = 0
        self.records = []  # 所有记录
        
    def evolve(self):
        """系统演化一步"""
        old_state = self.states[self.current_time].copy()
        old_descs = self.descriptions[self.current_time].copy()
        
        # 创建新状态（确保不同）
        new_element = f"elem_t{self.current_time+1}"
        new_state = old_state | {new_element}
        
        # 创建新描述
        new_descs = old_descs.copy()
        new_descs[new_element] = f"desc_of_{new_element}"
        
        # 记录转换
        transition = StateTransition(
            frozenset(old_state),
            frozenset(new_state),
            self.current_time
        )
        self.transitions.append(transition)
        
        # 更新信息集合
        new_info = self.information[-1].copy()
        new_info.add(InformationRecord(
            content=transition,
            source='evolution',
            timestamp=self.current_time
        ))
        
        # 添加到序列
        self.states.append(new_state)
        self.descriptions.append(new_descs)
        self.information.append(new_info)
        
        self.current_time += 1
        
    def create_observer(self):
        """创建观察者"""
        obs_id = f"obs_{len(self.observers)}"
        observer = Observer(self, obs_id)
        self.observers.append(observer)
        
        # 观察者也是系统的一部分
        current_state = self.states[self.current_time]
        current_state.add(observer)
        
        # 添加观察者描述
        current_descs = self.descriptions[self.current_time]
        current_descs[observer] = f"observer_{obs_id}"
        
        return observer
        
    def get_state(self, time):
        """获取特定时刻的状态"""
        if 0 <= time < len(self.states):
            return self.states[time]
        return None
        
    def calculate_entropy(self, time=None):
        """计算系统熵"""
        if time is None:
            time = self.current_time
            
        if 0 <= time < len(self.descriptions):
            desc_set = set(self.descriptions[time].values())
            return math.log2(len(desc_set)) if desc_set else 0
        return 0
        
    def time_metric(self, i, j):
        """时间度量函数"""
        if i == j:
            return 0.0
            
        if i > j:
            i, j = j, i  # 确保i < j
            
        total_diff = 0.0
        for k in range(i, j):
            if k < len(self.states) - 1:
                state_k = self.states[k]
                state_k1 = self.states[k+1]
                diff = len(state_k1 - state_k)
                total_diff += diff
                
        return total_diff
        
    def get_information(self, time=None):
        """获取信息集合"""
        if time is None:
            time = self.current_time
            
        if 0 <= time < len(self.information):
            return self.information[time]
        return set()
        
    def get_observer(self):
        """获取活跃观察者"""
        if self.observers:
            return self.observers[0]
        return None
        
    def get_all_elements(self):
        """获取系统中所有元素"""
        all_elements = set()
        for state in self.states:
            all_elements.update(state)
        return all_elements
        
    def get_observable_element(self):
        """获取可观察元素"""
        current_state = self.states[self.current_time]
        # 返回非观察者元素
        for elem in current_state:
            if not isinstance(elem, Observer):
                return elem
        return 'default_target'
        
    def get_current_state(self):
        """获取当前状态"""
        return self.states[self.current_time]
        
    def add_record(self, record):
        """添加记录到系统"""
        self.records.append(record)
        current_state = self.states[self.current_time]
        current_state.add(record)
        
        # 更新描述
        current_descs = self.descriptions[self.current_time]
        current_descs[record] = f"record_{len(self.records)}"
        
        # 记录也是信息
        info = InformationRecord(
            content=record,
            source='observation',
            timestamp=self.current_time
        )
        self.information[self.current_time].add(info)
        
    def check_entropy_increase(self):
        """检查熵增条件"""
        if len(self.states) < 2:
            return False  # 需要至少两个状态才能判断熵增
            
        for i in range(len(self.states)-1):
            entropy_i = self.calculate_entropy(i)
            entropy_i1 = self.calculate_entropy(i+1)
            if entropy_i1 <= entropy_i:
                return False
        return True
        
    def check_state_asymmetry(self):
        """检查状态不对称"""
        if len(self.states) < 2:
            return False  # 需要至少两个状态
            
        for i in range(len(self.states)-1):
            if self.states[i] == self.states[i+1]:
                return False
        return True
        
    def check_time_existence(self):
        """检查时间存在性"""
        if len(self.states) < 2:
            return False
            
        # 验证时间度量的性质
        for i in range(min(3, len(self.states))):
            for j in range(i, min(3, len(self.states))):
                tau = self.time_metric(i, j)
                if i == j and tau != 0:
                    return False
                if i < j and tau <= 0:
                    return False
        return True
        
    def check_information_emergence(self):
        """检查信息涌现"""
        if len(self.information) < 2:
            return False  # 需要至少两个时刻
            
        # 检查单调性
        for i in range(len(self.information)-1):
            if not self.information[i].issubset(self.information[i+1]):
                return False
        return True
        
    def check_observer_existence(self):
        """检查观察者存在"""
        if not self.observers:
            return False
            
        # 验证观察者在系统内
        all_elements = self.get_all_elements()
        for obs in self.observers:
            if obs not in all_elements:
                return False
        return True
        
    def verify_all_conditions(self):
        """验证所有条件"""
        return {
            'entropy': self.check_entropy_increase(),
            'asymmetry': self.check_state_asymmetry(),
            'time': self.check_time_existence(),
            'information': self.check_information_emergence(),
            'observer': self.check_observer_existence()
        }


class TestT1_2_FiveFoldEquivalence(VerificationTest):
    """T1-2 五重等价性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        
    def test_entropy_implies_asymmetry(self):
        """测试熵增蕴含不对称 - 验证检查点1"""
        system = FiveFoldSystem()
        
        # 演化系统确保熵增
        for _ in range(5):
            system.evolve()
            
        # 验证熵确实在增加
        self.assertTrue(
            system.check_entropy_increase(),
            "System should have entropy increase"
        )
        
        # 验证状态不对称
        self.assertTrue(
            system.check_state_asymmetry(),
            "Entropy increase should imply state asymmetry"
        )
        
        # 反向测试：如果强制状态相同
        # 创建新系统
        static_system = FiveFoldSystem()
        # 保持状态不变
        for _ in range(3):
            static_system.states.append(static_system.states[-1].copy())
            static_system.descriptions.append(static_system.descriptions[-1].copy())
            static_system.information.append(static_system.information[-1].copy())
            
        # 验证没有熵增
        self.assertFalse(
            static_system.check_entropy_increase(),
            "Static states should not have entropy increase"
        )
        
    def test_asymmetry_implies_time(self):
        """测试不对称定义时间 - 验证检查点2"""
        system = FiveFoldSystem()
        
        # 演化产生不对称状态序列
        for _ in range(5):
            system.evolve()
            
        # 验证状态不对称
        self.assertTrue(
            system.check_state_asymmetry(),
            "System should have asymmetric states"
        )
        
        # 验证时间度量存在且有效
        self.assertTrue(
            system.check_time_existence(),
            "Asymmetric states should define time metric"
        )
        
        # 测试时间度量性质
        # 非负性和同一性
        for i in range(3):
            self.assertEqual(
                system.time_metric(i, i), 0.0,
                f"τ({i},{i}) should be 0"
            )
            
        # 正定性
        for i in range(3):
            for j in range(i+1, 4):
                self.assertGreater(
                    system.time_metric(i, j), 0,
                    f"τ({i},{j}) should be positive"
                )
                
        # 可加性
        tau_02 = system.time_metric(0, 2)
        tau_01 = system.time_metric(0, 1)
        tau_12 = system.time_metric(1, 2)
        self.assertAlmostEqual(
            tau_02, tau_01 + tau_12,
            msg="Time metric should be additive"
        )
        
    def test_time_implies_information(self):
        """测试时间产生信息 - 验证检查点3"""
        system = FiveFoldSystem()
        
        # 演化产生时间流逝
        for _ in range(5):
            system.evolve()
            
        # 验证时间存在
        self.assertTrue(
            system.check_time_existence(),
            "System should have time structure"
        )
        
        # 验证信息涌现
        self.assertTrue(
            system.check_information_emergence(),
            "Time should produce information"
        )
        
        # 验证信息单调增长
        info_sizes = [len(system.get_information(t)) for t in range(len(system.information))]
        for i in range(len(info_sizes)-1):
            self.assertGreaterEqual(
                info_sizes[i+1], info_sizes[i],
                f"Information should grow monotonically at t={i}"
            )
            
        # 验证信息包含转换记录
        final_info = system.get_information()
        has_transitions = any(
            isinstance(info.content, StateTransition) 
            for info in final_info
        )
        self.assertTrue(
            has_transitions,
            "Information should contain state transitions"
        )
        
    def test_information_implies_observer(self):
        """测试信息需要观察者 - 验证检查点4"""
        system = FiveFoldSystem()
        
        # 演化产生信息
        for _ in range(3):
            system.evolve()
            
        # 验证信息存在
        self.assertTrue(
            system.check_information_emergence(),
            "System should have information"
        )
        
        # 创建观察者来处理信息
        observer = system.create_observer()
        
        # 验证观察者存在
        self.assertTrue(
            system.check_observer_existence(),
            "Information should require observer"
        )
        
        # 验证观察者能处理信息
        info = system.get_information()
        result = observer.process_information(info)
        self.assertIsNotNone(
            result,
            "Observer should be able to process information"
        )
        
        # 验证观察者是内生的
        self.assertIn(
            observer, system.get_all_elements(),
            "Observer should be part of the system"
        )
        
    def test_observer_implies_entropy(self):
        """测试观察产生熵增 - 验证检查点5"""
        system = FiveFoldSystem()
        
        # 创建观察者
        observer = system.create_observer()
        
        # 进行多次观察
        for _ in range(5):
            # 记录观察前的熵
            initial_entropy = system.calculate_entropy()
            initial_state_size = len(system.get_current_state())
            
            # 执行观察
            target = system.get_observable_element()
            record = observer.observe(target)
            
            # 将记录添加到系统
            system.add_record(record)
            
            # 验证状态增长
            final_state_size = len(system.get_current_state())
            self.assertGreater(
                final_state_size, initial_state_size,
                "Observation should increase state size"
            )
            
            # 计算新熵
            final_entropy = system.calculate_entropy()
            self.assertGreater(
                final_entropy, initial_entropy,
                "Observation should increase entropy"
            )
            
            # 准备下一轮
            system.evolve()
            
    def test_full_equivalence(self):
        """测试完整的五重等价性"""
        system = FiveFoldSystem()
        
        # 初始状态：所有条件都不满足（除了平凡情况）
        initial_conditions = system.verify_all_conditions()
        
        # 创建完整的自指系统
        system.create_observer()
        for _ in range(5):
            system.evolve()
            
            # 观察者观察
            if system.observers:
                obs = system.observers[0]
                target = system.get_observable_element()
                record = obs.observe(target)
                system.add_record(record)
                
        # 验证所有条件
        final_conditions = system.verify_all_conditions()
        
        # 所有条件应该同时为真
        all_true = all(final_conditions.values())
        self.assertTrue(
            all_true,
            f"All conditions should be true: {final_conditions}"
        )
        
        # 测试等价性：如果一个为真，全部为真
        self.assertEqual(
            len(set(final_conditions.values())), 1,
            "All conditions should have the same truth value"
        )
        
    def test_condition_correlation(self):
        """测试条件间的相关性"""
        # 创建多个系统，逐步满足不同条件
        systems = []
        
        # 系统1：仅演化（应该满足所有条件）
        s1 = FiveFoldSystem()
        for _ in range(5):
            s1.evolve()
        s1.create_observer()
        systems.append(('evolved', s1))
        
        # 系统2：静态系统（不满足任何条件）
        s2 = FiveFoldSystem()
        # 不演化
        systems.append(('static', s2))
        
        # 验证相关性
        for name, system in systems:
            conditions = system.verify_all_conditions()
            values = list(conditions.values())
            
            # 所有条件应该一致
            self.assertTrue(
                all(v == values[0] for v in values),
                f"{name}: Conditions should be correlated: {conditions}"
            )
            
    def test_minimal_system(self):
        """测试最小系统的等价性"""
        system = FiveFoldSystem()
        
        # 最小演化：只添加一个元素
        system.evolve()
        
        # 检查条件
        conditions = system.verify_all_conditions()
        
        # 即使是最小变化，也应该满足基本等价性
        # （观察者可能还不存在）
        self.assertEqual(
            conditions['entropy'], conditions['asymmetry'],
            "Entropy and asymmetry should be equivalent"
        )
        
        self.assertEqual(
            conditions['asymmetry'], conditions['time'],
            "Asymmetry and time should be equivalent"
        )
        
    def test_observer_observation_cycle(self):
        """测试观察者观察循环"""
        system = FiveFoldSystem()
        
        # 创建两个观察者
        obs1 = system.create_observer()
        obs2 = system.create_observer()
        
        # 交替观察
        for i in range(3):
            # obs1观察系统
            target1 = system.get_observable_element()
            record1 = obs1.observe(target1)
            system.add_record(record1)
            
            # obs2观察obs1的记录
            record2 = obs2.observe(record1)
            system.add_record(record2)
            
            system.evolve()
            
        # 验证熵持续增加
        self.assertTrue(
            system.check_entropy_increase(),
            "Observation cycle should maintain entropy increase"
        )
        
        # 验证所有等价条件
        conditions = system.verify_all_conditions()
        self.assertTrue(
            all(conditions.values()),
            "All equivalence conditions should hold"
        )
        
    def test_information_accumulation(self):
        """测试信息累积过程"""
        system = FiveFoldSystem()
        
        info_history = []
        
        # 演化并记录信息
        for t in range(5):
            info_t = system.get_information(t)
            info_history.append(len(info_t))
            system.evolve()
            
        # 验证信息单调增长
        for i in range(len(info_history)-1):
            self.assertGreaterEqual(
                info_history[i+1], info_history[i],
                f"Information should accumulate: {info_history}"
            )
            
        # 验证信息内容
        final_info = system.get_information()
        
        # 应该包含转换和其他记录
        info_types = set()
        for info in final_info:
            if isinstance(info.content, StateTransition):
                info_types.add('transition')
            else:
                info_types.add('other')
                
        self.assertIn(
            'transition', info_types,
            "Information should contain transitions"
        )


if __name__ == "__main__":
    unittest.main()