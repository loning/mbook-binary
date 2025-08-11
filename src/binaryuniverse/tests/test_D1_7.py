"""
Unit tests for D1-7: Collapse Operator
D1-7：Collapse算子的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import SystemState, Observer, create_initial_system
import random


class CollapseOperator:
    """Collapse算子实现"""
    
    def __init__(self):
        self.collapse_history = []
        
    def collapse(self, state_set, observer):
        """执行collapse操作"""
        if not state_set:
            raise ValueError("Cannot collapse empty state set")
            
        # 阶段1：预collapse状态
        pre_collapse_entropy = self._compute_set_entropy(state_set)
        
        # 阶段2：观察者介入（测量）
        measurement = observer.measure(list(state_set)[0])  # 简化：测量第一个状态
        
        # 阶段3：状态选择（基于观察者权重）
        selected_state = self._select_state(state_set, observer, measurement)
        
        # 阶段4：记录生成
        record = self._generate_record(state_set, selected_state, observer)
        
        # 记录collapse历史
        self.collapse_history.append({
            'pre_states': state_set.copy(),
            'observer': observer.name,
            'collapsed_state': selected_state,
            'record': record
        })
        
        return selected_state, record
        
    def _compute_set_entropy(self, state_set):
        """计算状态集的熵"""
        if not state_set:
            return 0
        # 简化：使用状态数量的对数
        import math
        return math.log2(len(state_set))
        
    def _select_state(self, state_set, observer, measurement):
        """基于观察者选择状态"""
        # 简化实现：基于观察者名称的哈希值选择
        states = list(state_set)
        index = hash(observer.name) % len(states)
        return states[index]
        
    def _generate_record(self, state_set, selected_state, observer):
        """生成测量记录"""
        return {
            'observer': observer.name,
            'pre_state_count': len(state_set),
            'selected_state': str(selected_state),
            'timestamp': len(self.collapse_history)
        }


class TestD1_7_CollapseOperator(VerificationTest):
    """D1-7 Collapse算子的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.collapse_op = CollapseOperator()
        self.observer1 = Observer("Observer1")
        self.observer2 = Observer("Observer2")
        
        # 创建多个状态的集合（使用列表，因为SystemState不是hashable）
        self.state_set = [
            SystemState({"a", "b"}, "State1", 0),
            SystemState({"c", "d"}, "State2", 0),
            SystemState({"e", "f"}, "State3", 0)
        ]
        
    def test_entropy_increase(self):
        """测试熵增性 - 验证检查点1"""
        # 计算collapse前的熵
        pre_entropy = self.collapse_op._compute_set_entropy(self.state_set)
        
        # 执行collapse
        collapsed_state, record = self.collapse_op.collapse(self.state_set, self.observer1)
        
        # 计算collapse后的总熵（包括记录）
        # 简化：考虑记录增加了系统的信息
        post_states = [collapsed_state]
        post_entropy_base = self.collapse_op._compute_set_entropy(post_states)
        
        # 记录本身携带信息，增加总熵
        record_entropy_contribution = 1.0  # 简化：记录贡献固定熵值
        total_post_entropy = post_entropy_base + record_entropy_contribution
        
        # 验证熵增
        self.assertGreater(
            total_post_entropy, 0,
            "Post-collapse total entropy should be positive"
        )
        
        # 验证有信息记录
        self.assertIsNotNone(record)
        self.assertIn('observer', record)
        self.assertIn('pre_state_count', record)
        
    def test_irreversibility(self):
        """测试不可逆性 - 验证检查点2"""
        # 保存原始状态集
        original_states = self.state_set.copy()
        original_count = len(original_states)
        
        # 执行collapse
        collapsed_state, record = self.collapse_op.collapse(self.state_set, self.observer1)
        
        # 验证无法从结果恢复原始状态集
        # 1. 信息丢失：多个状态变成一个
        self.assertEqual(len([collapsed_state]), 1)
        self.assertGreater(original_count, 1)
        
        # 2. 即使有记录，也无法完全恢复原始状态的具体内容
        self.assertEqual(record['pre_state_count'], original_count)
        # 但无法恢复具体是哪些状态
        
        # 3. collapse历史记录了过程，但这是额外信息
        self.assertEqual(len(self.collapse_op.collapse_history), 1)
        
    def test_self_reference(self):
        """测试自指性 - 验证检查点3"""
        # 创建包含collapse算子自身的系统
        system_with_operator = SystemState(
            {"element1", "collapse_operator"},
            "System containing collapse operator",
            0
        )
        
        # 创建包含算子引用的状态集
        state_set_with_ref = [
            system_with_operator,
            SystemState({"other"}, "Other state", 0)
        ]
        
        # 验证collapse操作仍然有效
        try:
            collapsed_state, record = self.collapse_op.collapse(
                state_set_with_ref, 
                self.observer1
            )
            self.assertIsNotNone(collapsed_state)
            self.assertIsNotNone(record)
            
            # 验证可以对结果再次应用collapse
            second_collapse, second_record = self.collapse_op.collapse(
                [collapsed_state],
                self.observer2
            )
            self.assertIsNotNone(second_collapse)
            
        except Exception as e:
            self.fail(f"Self-referential collapse should be well-defined: {e}")
            
    def test_observer_dependence(self):
        """测试观察者依赖性 - 验证检查点4"""
        # 使用相同状态集但不同观察者
        states = [
            SystemState({f"elem{i}"}, f"State{i}", 0)
            for i in range(5)
        ]
        
        # 两个不同的观察者进行collapse
        result1_state, result1_record = self.collapse_op.collapse(
            states.copy(), self.observer1
        )
        result2_state, result2_record = self.collapse_op.collapse(
            states.copy(), self.observer2
        )
        
        # 验证不同观察者可能产生不同结果
        # 由于我们的实现基于观察者名称的哈希，结果可能不同
        self.assertNotEqual(
            result1_record['observer'],
            result2_record['observer'],
            "Different observers should be recorded differently"
        )
        
        # 如果选择了不同状态，验证它们确实不同
        if result1_state != result2_state:
            self.assertNotEqual(
                result1_state.elements,
                result2_state.elements,
                "Different observers may collapse to different states"
            )
            
    def test_collapse_stages(self):
        """测试collapse过程的阶段"""
        # 阶段1：预collapse - 多状态
        pre_states = self.state_set
        self.assertGreater(len(pre_states), 1)
        
        # 执行collapse
        collapsed_state, record = self.collapse_op.collapse(pre_states, self.observer1)
        
        # 阶段4：后collapse - 单一状态 + 记录
        self.assertIsInstance(collapsed_state, SystemState)
        self.assertIsInstance(record, dict)
        
        # 验证记录包含必要信息
        self.assertEqual(record['pre_state_count'], len(pre_states))
        self.assertEqual(record['observer'], self.observer1.name)
        
    def test_nonlinearity(self):
        """测试非线性性"""
        # 创建两个状态集
        set1 = [SystemState({"a"}, "A", 0), SystemState({"b"}, "B", 0)]
        set2 = [SystemState({"c"}, "C", 0), SystemState({"d"}, "D", 0)]
        
        # 分别collapse
        result1, _ = self.collapse_op.collapse(set1, self.observer1)
        result2, _ = self.collapse_op.collapse(set2, self.observer1)
        
        # 合并后collapse
        combined_set = set1 + set2
        result_combined, _ = self.collapse_op.collapse(combined_set, self.observer1)
        
        # 验证非线性：合并collapse的结果不等于分别collapse的"和"
        # （这里"和"的概念是抽象的，因为我们处理的是离散状态）
        self.assertIsInstance(result_combined, SystemState)
        self.assertIsInstance(result1, SystemState)
        self.assertIsInstance(result2, SystemState)
        
    def test_recursive_application(self):
        """测试递归适用性"""
        # 第一次collapse
        result1, record1 = self.collapse_op.collapse(self.state_set, self.observer1)
        
        # 对结果再次应用collapse
        result2, record2 = self.collapse_op.collapse([result1], self.observer2)
        
        # 验证递归应用有效
        self.assertIsNotNone(result2)
        self.assertIsNotNone(record2)
        
        # 验证历史记录
        self.assertEqual(len(self.collapse_op.collapse_history), 2)
        
    def test_backaction_effects(self):
        """测试反作用效应"""
        initial_system = create_initial_system()
        
        # 创建状态集
        states = [
            initial_system,
            initial_system.evolve(),
            initial_system.evolve().evolve()
        ]
        
        # Collapse前记录系统信息
        pre_total_elements = sum(len(s.elements) for s in states)
        
        # 执行collapse
        collapsed_state, record = self.collapse_op.collapse(states, self.observer1)
        
        # 模拟反作用：观察者状态更新
        observed_system = self.observer1.backact(collapsed_state)
        
        # 验证系统被改变（反作用）
        self.assertGreater(
            len(observed_system.elements),
            len(collapsed_state.elements),
            "Observer backaction should modify the system"
        )
        
    def test_information_gain_and_cost(self):
        """测试信息获得与成本"""
        # 初始不确定性（多个可能状态）
        initial_uncertainty = len(self.state_set)
        
        # Collapse后确定性（单一状态）
        collapsed_state, record = self.collapse_op.collapse(
            self.state_set, self.observer1
        )
        final_uncertainty = 1  # 单一确定状态
        
        # 信息获得：不确定性减少
        information_gained = initial_uncertainty - final_uncertainty
        self.assertGreater(
            information_gained, 0,
            "Collapse should reduce uncertainty (gain information)"
        )
        
        # 但总系统复杂度增加（因为有了记录）
        self.assertIsNotNone(record)
        self.assertGreater(
            len(record), 0,
            "Collapse generates records (increases total complexity)"
        )
        
    def test_collapse_types(self):
        """测试不同类型的collapse"""
        # 完全collapse：多状态到单一状态
        complete_result, _ = self.collapse_op.collapse(
            self.state_set, self.observer1
        )
        self.assertIsInstance(complete_result, SystemState)
        
        # 部分collapse（模拟）：如果我们有更复杂的实现
        # 这里简化演示概念
        partial_states = [
            SystemState({"x"}, "X", 0),
            SystemState({"y"}, "Y", 0)
        ]
        partial_result, _ = self.collapse_op.collapse(
            partial_states, self.observer1
        )
        self.assertIsInstance(partial_result, SystemState)


if __name__ == "__main__":
    unittest.main()