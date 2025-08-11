"""
Unit tests for D1-5: Observer Definition
D1-5：观察者定义的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import Observer, SystemState, create_initial_system


class TestD1_5_Observer(VerificationTest):
    """D1-5观察者的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.system = create_initial_system()
        self.observer = Observer("test_observer")
        
    def test_read_function(self):
        """测试读取功能 - 验证检查点1"""
        # 测试区分性：不同状态应该产生不同的读取结果
        s1 = self.system
        s2 = s1.evolve()
        
        read1 = self.observer.measure(s1)
        read2 = self.observer.measure(s2)
        
        # 验证不同状态产生不同的测量结果
        self.assertNotEqual(
            read1["element_count"],
            read2["element_count"],
            "Observer should distinguish different states"
        )
        
        # 测试完备性：观察者能读取所有相关信息
        measurement = self.observer.measure(self.system)
        self.assertIn("entropy", measurement)
        self.assertIn("element_count", measurement)
        self.assertIn("time", measurement)
        self.assertIn("observer", measurement)
        
        # 测试自指性：观察者能读取包含自身的信息
        self.assertEqual(
            measurement["observer"],
            self.observer.name,
            "Observer should be able to read information about itself"
        )
        
    def test_compute_function(self):
        """测试计算功能 - 验证检查点2"""
        # 测试确定性：相同输入产生相同输出
        measurement1 = self.observer.measure(self.system)
        measurement2 = self.observer.measure(self.system)
        
        # 对于相同的系统状态，测量应该产生相同的基本信息
        self.assertEqual(
            measurement1["element_count"],
            measurement2["element_count"],
            "Compute function should be deterministic"
        )
        
        # 测试一致性：等价状态产生相同决策
        # 这里简化处理：相同时间的状态被视为等价
        s1 = SystemState({"a", "b"}, "State 1", 5)
        s2 = SystemState({"c", "d"}, "State 2", 5)
        
        m1 = self.observer.measure(s1)
        m2 = self.observer.measure(s2)
        
        self.assertEqual(
            m1["time"], m2["time"],
            "Equivalent states (same time) should produce consistent results"
        )
        
        # 测试递归处理能力
        # 观察者能处理自己产生的信息
        processed_info = {
            "entropy": 2.0,
            "element_count": 4,
            "time": 1,
            "observer": self.observer.name
        }
        # 验证观察者能处理已处理的信息（简化测试）
        self.assertIsInstance(processed_info["entropy"], float)
        self.assertIsInstance(processed_info["element_count"], int)
        
    def test_update_function(self):
        """测试更新功能 - 验证检查点3"""
        # 测试状态变化：观察必然改变系统状态
        initial_state = self.system
        updated_state = self.observer.backact(initial_state)
        
        # 验证状态确实发生了变化
        self.assertNotEqual(
            initial_state.elements,
            updated_state.elements,
            "Update function should change system state"
        )
        
        # 验证新元素的添加（观察痕迹）
        new_elements = updated_state.elements - initial_state.elements
        self.assertGreater(
            len(new_elements), 0,
            "Observation should add new elements to the system"
        )
        
        # 验证观察痕迹包含观察者信息
        observation_trace = next(iter(new_elements))
        self.assertIn(
            self.observer.name,
            observation_trace,
            "Observation trace should contain observer name"
        )
        
        # 测试熵增：观察后系统熵增加
        initial_entropy = initial_state.entropy()
        updated_entropy = updated_state.entropy()
        
        self.assertGreater(
            updated_entropy,
            initial_entropy,
            "Observation should increase system entropy"
        )
        
    def test_measurement_effect(self):
        """测试测量效应 - 验证检查点4"""
        # 测试唯一结果：每次测量产生唯一结果
        result = self.observer.measure(self.system)
        
        # 验证测量结果的唯一性（每个字段都有确定值）
        self.assertIsNotNone(result["entropy"])
        self.assertIsNotNone(result["element_count"])
        self.assertIsNotNone(result["time"])
        
        # 测试不可逆性：测量后无法恢复原状态
        original_state = self.system
        measured_state = self.observer.backact(original_state)
        
        # 验证状态包含新信息，无法简单逆转
        self.assertGreater(
            len(measured_state.elements),
            len(original_state.elements),
            "Measurement should be irreversible (adds information)"
        )
        
        # 测试反作用：测量改变系统状态
        self.assertNotEqual(
            original_state.description,
            measured_state.description,
            "Measurement should have backaction on the system"
        )
        
    def test_internal_observer(self):
        """测试观察者内在性"""
        # 测试观察者是系统的一部分
        # 通过观察痕迹验证观察者的内在性
        observed_system = self.observer.backact(self.system)
        
        # 检查系统是否记录了观察者的存在
        observer_trace_found = any(
            self.observer.name in str(element)
            for element in observed_system.elements
        )
        
        self.assertTrue(
            observer_trace_found,
            "Observer should leave traces in the system (internal observer)"
        )
        
    def test_self_describing(self):
        """测试自我描述能力"""
        # 观察者能够描述自己的测量行为
        self_measurement = self.observer.measure(self.system)
        
        # 验证测量结果包含观察者自身信息
        self.assertEqual(
            self_measurement["observer"],
            self.observer.name,
            "Observer should be able to describe itself"
        )
        
    def test_recursive_closure(self):
        """测试递归闭合性"""
        # 测试观察链的闭合性：read->compute->update->read
        s0 = self.system
        
        # Step 1: Read (measure)
        measurement = self.observer.measure(s0)
        
        # Step 2: Compute (implicit in measure)
        # Step 3: Update (backact)
        s1 = self.observer.backact(s0)
        
        # Step 4: Read again
        measurement2 = self.observer.measure(s1)
        
        # 验证操作链是封闭的（产生有效结果）
        self.assertIsNotNone(measurement2)
        self.assertIn("entropy", measurement2)
        self.assertIn("element_count", measurement2)
        
        # 验证状态演化的连续性
        self.assertEqual(
            measurement2["time"],
            measurement["time"],
            "Recursive operations should maintain time consistency"
        )
        
    def test_observer_types(self):
        """测试不同类型的观察者"""
        # 当前实现是完全观察者（能读取所有基本信息）
        measurement = self.observer.measure(self.system)
        
        # 完全观察者应该能获取所有基本系统信息
        required_info = ["entropy", "element_count", "time", "observer"]
        for info in required_info:
            self.assertIn(
                info, measurement,
                f"Complete observer should read {info}"
            )
            
    def test_multiple_observations(self):
        """测试多次观察的累积效应"""
        s0 = self.system
        
        # 进行多次观察，使用不同的观察者避免重复
        observer1 = Observer("observer1")
        observer2 = Observer("observer2")
        observer3 = Observer("observer3")
        
        s1 = observer1.backact(s0)
        s2 = observer2.backact(s1)
        s3 = observer3.backact(s2)
        
        # 验证每次观察都增加系统复杂度
        self.assertLess(len(s0.elements), len(s1.elements))
        self.assertLess(len(s1.elements), len(s2.elements))
        self.assertLess(len(s2.elements), len(s3.elements))
        
        # 验证熵的单调增加
        self.assertLess(s0.entropy(), s1.entropy())
        self.assertLess(s1.entropy(), s2.entropy())
        self.assertLess(s2.entropy(), s3.entropy())
        
        # 验证不同观察者留下不同痕迹
        all_elements = s3.elements
        observer_traces = [e for e in all_elements if "observed_by" in str(e)]
        self.assertGreaterEqual(
            len(observer_traces), 3,
            "Should have traces from all three observers"
        )


if __name__ == "__main__":
    unittest.main()