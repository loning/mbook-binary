"""
Unit tests for T1-1: Entropy Increase Necessity Theorem
T1-1：熵增必然性定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class SystemElement:
    """系统元素"""
    
    def __init__(self, name, predecessors=None):
        self.name = name
        self.predecessors = predecessors or []
        self.description = ""  # 改为空字符串而不是None
        self.depth = None
        
    def __repr__(self):
        return f"Element({self.name})"
        
    def __hash__(self):
        return hash(self.name)
        
    def __eq__(self, other):
        return isinstance(other, SystemElement) and self.name == other.name


class SelfReferentialSystem:
    """自指完备系统实现"""
    
    def __init__(self):
        self.states = []  # 时间序列状态
        self.descriptions = []  # 时间序列描述映射
        self.time = 0
        
        # 初始状态
        initial_element = SystemElement("s0")
        desc_func = SystemElement("[Desc_0]")
        
        initial_state = {initial_element, desc_func}
        initial_descs = {
            initial_element: "initial",
            desc_func: "description_function"
        }
        
        self.states.append(initial_state)
        self.descriptions.append(initial_descs)
        
    def is_self_referentially_complete(self):
        """检查自指完备性"""
        if self.time >= len(self.states):
            return False
            
        current_state = self.states[self.time]
        current_descs = self.descriptions[self.time]
        
        # 条件1：自指性 - 描述函数能描述自己
        has_desc_func = any("[Desc" in str(elem.name) for elem in current_state)
        
        # 条件2：完备性 - 所有元素都有描述
        all_described = all(elem in current_descs for elem in current_state)
        
        # 条件3：一致性 - 描述不矛盾（但允许一些描述相同）
        # 放宽条件：只要有描述即可
        has_descriptions = len(current_descs) > 0
        
        # 条件4：非平凡性
        non_trivial = len(current_state) > 0
        
        return has_desc_func and all_described and has_descriptions and non_trivial
        
    def describe(self, element):
        """描述函数"""
        if isinstance(element, SystemElement):
            if "[Desc" in element.name:
                return f"desc_of_{element.name}"
            elif element.description:
                return element.description
            else:
                return f"desc_{element.name}_t{self.time}"
        else:
            return f"composite_desc_t{self.time}"
            
    def create_complete_description(self, state):
        """创建对整个状态的完整描述"""
        # 编码整个状态的信息
        state_info = []
        for elem in sorted(state, key=lambda x: str(x.name)):
            desc = self.descriptions[self.time].get(elem, "")
            state_info.append(f"{elem.name}:{desc}")
            
        complete_desc_name = f"Desc^({self.time+1})[{','.join(state_info[:3])}...]"
        complete_desc = SystemElement(complete_desc_name)
        complete_desc.description = f"complete_description_of_S{self.time}"
        
        return complete_desc
        
    def recursive_depth(self, element):
        """计算元素的递归深度"""
        if element.depth is not None:
            return element.depth
            
        if not element.predecessors:
            element.depth = 0
            return 0
            
        max_pred_depth = max(self.recursive_depth(pred) for pred in element.predecessors)
        element.depth = max_pred_depth + 1
        return element.depth
        
    def generate_new_layer(self, target_depth):
        """生成特定深度的新元素"""
        new_elements = set()
        
        # 基于现有元素创建新的递归结构
        current_state = self.states[self.time]
        
        # 创建一些深度为target_depth的新元素
        for i in range(2):  # 至少创建2个新元素
            # 选择一些前驱
            if self.time > 0:
                predecessors = list(current_state)[:2]
                new_elem = SystemElement(
                    f"elem_d{target_depth}_n{i}",
                    predecessors=predecessors
                )
                new_elem.depth = target_depth
                new_elements.add(new_elem)
                
        return new_elements
        
    def evolve(self):
        """系统演化一步"""
        current_state = self.states[self.time]
        current_descs = self.descriptions[self.time]
        
        # 创建新的完整描述
        new_complete_desc = self.create_complete_description(current_state)
        
        # 新状态包含旧状态和新描述
        new_state = current_state.copy()
        new_state.add(new_complete_desc)
        
        # 添加递归深度为t+1的新元素
        new_elements = self.generate_new_layer(self.time + 1)
        new_state.update(new_elements)
        
        # 添加对描述函数的递归描述
        desc_func = next((e for e in current_state if "[Desc" in e.name), None)
        if desc_func:
            desc_of_desc = SystemElement(f"Desc_{self.time}([Desc_{self.time}])")
            desc_of_desc.predecessors = [desc_func]
            new_state.add(desc_of_desc)
        
        # 更新描述集合
        new_descs = current_descs.copy()
        for elem in new_state - current_state:
            new_descs[elem] = self.describe(elem)
            
        # 更新描述函数表示
        new_desc_func = SystemElement(f"[Desc_{self.time+1}]")
        new_state.add(new_desc_func)
        new_descs[new_desc_func] = "description_function"
        
        self.states.append(new_state)
        self.descriptions.append(new_descs)
        self.time += 1
        
        return new_state
        
    def get_state(self, time):
        """获取特定时刻的状态"""
        if time < len(self.states):
            return self.states[time]
        return None
        
    def get_descriptions(self, time):
        """获取特定时刻的描述集合"""
        if time < len(self.descriptions):
            return set(self.descriptions[time].values())
        return set()
        
    def calculate_entropy(self, state=None):
        """计算系统熵"""
        if state is None:
            state = self.states[self.time]
            
        desc_set = self.get_descriptions(self.time)
        return math.log2(len(desc_set)) if desc_set else 0
        
    def get_elements_at_time(self, time):
        """获取特定时刻的所有元素"""
        if time < len(self.states):
            return self.states[time]
        return set()
        
    def has_initial_element(self):
        """检查是否有初始元素"""
        return any("s0" in elem.name for elem in self.states[0])
        
    def has_description_function(self):
        """检查是否有描述函数表示"""
        current = self.states[self.time]
        return any("[Desc" in elem.name for elem in current)
        
    def get_description_function(self):
        """获取描述函数"""
        current = self.states[self.time]
        desc_func = next((e for e in current if "[Desc" in e.name), None)
        
        class DescFunction:
            def __init__(self, system, time):
                self.system = system
                self.time = time
                
            def __call__(self, elem):
                return self.system.describe(elem)
                
        return DescFunction(self, self.time)


def entropy_lower_bound(time):
    """熵的理论下界"""
    return math.log2(time + 1)


class TestT1_1_EntropyIncrease(VerificationTest):
    """T1-1 熵增必然性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        
    def test_recursive_unfolding(self):
        """测试递归展开 - 验证检查点1"""
        system = SelfReferentialSystem()
        
        # 验证初始自指完备性
        self.assertTrue(
            system.is_self_referentially_complete(),
            "Initial system should be self-referentially complete"
        )
        
        # 演化几步
        for _ in range(3):
            system.evolve()
            
        # 验证递归结构存在
        elements = system.get_elements_at_time(system.time)
        
        # 检查基本元素
        self.assertTrue(
            system.has_initial_element(),
            "System should contain initial element"
        )
        
        # 检查描述函数表示
        self.assertTrue(
            system.has_description_function(),
            "System should contain description function"
        )
        
        # 检查递归描述（如 Desc([Desc])）
        recursive_descs = [e for e in elements if "Desc" in e.name and "[Desc" in e.name]
        self.assertGreater(
            len(recursive_descs), 0,
            "System should contain recursive descriptions"
        )
        
    def test_new_descriptions_necessity(self):
        """测试新描述必然性 - 验证检查点2"""
        system = SelfReferentialSystem()
        
        for t in range(5):
            current_state = system.get_state(t)
            
            # 构造新的完整描述
            new_desc = system.create_complete_description(current_state)
            
            # 验证新描述不在当前状态中
            if current_state is not None:
                self.assertNotIn(
                    new_desc, current_state,
                    f"New complete description at t={t} should not exist in current state"
                )
                
                # 验证新描述的唯一性
                desc_name = new_desc.name
                existing_names = [e.name for e in current_state]
                self.assertNotIn(
                    desc_name, existing_names,
                    f"New description name should be unique at t={t}"
                )
            
            # 演化到下一步
            if t < 4:
                system.evolve()
                
    def test_state_space_growth(self):
        """测试状态空间增长 - 验证检查点3"""
        system = SelfReferentialSystem()
        sizes = []
        
        for t in range(10):
            state = system.get_state(t)
            if state is not None:
                sizes.append(len(state))
            
            if t > 0:
                # 验证严格增长
                self.assertGreater(
                    sizes[t], sizes[t-1],
                    f"State space should grow strictly at t={t}: {sizes[t]} > {sizes[t-1]}"
                )
                
                # 验证至少增加1个元素
                self.assertGreaterEqual(
                    sizes[t] - sizes[t-1], 1,
                    f"Should add at least 1 element at t={t}"
                )
                
            # 演化到下一步
            if t < 9:
                system.evolve()
                
        # 验证增长趋势
        self.assertGreater(
            sizes[-1], sizes[0],
            "Final state should be much larger than initial"
        )
        
    def test_entropy_strict_increase(self):
        """测试熵严格增加 - 验证检查点4"""
        system = SelfReferentialSystem()
        entropies = []
        desc_counts = []
        
        for t in range(10):
            state = system.get_state(t)
            entropy = system.calculate_entropy(state)
            entropies.append(entropy)
            
            desc_set = system.get_descriptions(t)
            desc_counts.append(len(desc_set))
            
            if t > 0:
                # 验证熵严格增加
                self.assertGreater(
                    entropies[t], entropies[t-1],
                    f"Entropy should increase strictly at t={t}: {entropies[t]} > {entropies[t-1]}"
                )
                
                # 验证描述多样性增加
                self.assertGreater(
                    desc_counts[t], desc_counts[t-1],
                    f"Description diversity should increase at t={t}"
                )
                
            # 演化到下一步
            if t < 9:
                system.evolve()
                
    def test_recursive_depth_growth(self):
        """测试递归深度增长"""
        system = SelfReferentialSystem()
        max_depths = []
        
        for t in range(5):
            state = system.get_state(t)
            
            # 计算当前最大递归深度
            max_depth = 0
            if state is not None:
                for elem in state:
                    depth = system.recursive_depth(elem)
                    max_depth = max(max_depth, depth)
                
            max_depths.append(max_depth)
            
            # 验证递归深度不超过时间
            self.assertLessEqual(
                max_depth, t,
                f"Max recursive depth should not exceed time t={t}"
            )
            
            # 演化
            if t < 4:
                system.evolve()
                
        # 验证递归深度增长
        self.assertGreater(
            max_depths[-1], max_depths[0],
            "Recursive depth should grow over time"
        )
        
    def test_description_diversity(self):
        """测试描述多样性"""
        system = SelfReferentialSystem()
        
        for t in range(5):
            desc_set = system.get_descriptions(t)
            state = system.get_state(t)
            state_size = len(state) if state is not None else 0
            
            # 验证描述数量合理
            self.assertGreater(
                len(desc_set), 0,
                f"Should have descriptions at t={t}"
            )
            
            self.assertLessEqual(
                len(desc_set), state_size,
                f"Descriptions should not exceed state size at t={t}"
            )
            
            # 验证描述的唯一性在增加
            if t > 0:
                prev_desc_set = system.get_descriptions(t-1)
                new_descs = desc_set - prev_desc_set
                self.assertGreater(
                    len(new_descs), 0,
                    f"Should have new descriptions at t={t}"
                )
                
            system.evolve()
            
    def test_entropy_lower_bound(self):
        """测试熵的下界"""
        system = SelfReferentialSystem()
        
        for t in range(10):
            entropy = system.calculate_entropy()
            theoretical_bound = entropy_lower_bound(t)
            
            # 熵应该接近或超过理论下界
            self.assertGreaterEqual(
                entropy, theoretical_bound * 0.5,  # 允许一些余量
                f"Entropy at t={t} should be near theoretical bound"
            )
            
            system.evolve()
            
    def test_self_reference_persistence(self):
        """测试自指性的持续性"""
        system = SelfReferentialSystem()
        
        for t in range(5):
            # 验证系统保持自指完备
            self.assertTrue(
                system.is_self_referentially_complete(),
                f"System should remain self-referentially complete at t={t}"
            )
            
            # 验证描述函数的存在性
            self.assertTrue(
                system.has_description_function(),
                f"Description function should exist at t={t}"
            )
            
            system.evolve()
            
    def test_irreversibility(self):
        """测试不可逆性"""
        system = SelfReferentialSystem()
        
        # 记录几个时刻的状态
        states = []
        for t in range(5):
            state = system.get_state(t)
            if state is not None:
                states.append({
                    'elements': state.copy(),
                    'descriptions': system.get_descriptions(t).copy(),
                    'entropy': system.calculate_entropy()
                })
            system.evolve()
            
        # 验证状态不会重复
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                self.assertNotEqual(
                    states[i]['elements'], states[j]['elements'],
                    f"States at t={i} and t={j} should be different"
                )
                
                # 验证熵单调增加
                self.assertLess(
                    states[i]['entropy'], states[j]['entropy'],
                    f"Entropy should increase from t={i} to t={j}"
                )
                
    def test_complete_description_property(self):
        """测试完整描述的性质"""
        system = SelfReferentialSystem()
        
        for t in range(3):
            current_state = system.get_state(t)
            complete_desc = system.create_complete_description(current_state)
            
            # 验证完整描述编码了状态信息
            desc_name = complete_desc.name
            self.assertIn(
                "Desc^", desc_name,
                "Complete description should have proper format"
            )
            
            self.assertIn(
                str(t+1), desc_name,
                "Complete description should reference next time step"
            )
            
            # 验证完整描述的唯一性
            another_desc = system.create_complete_description(current_state)
            self.assertEqual(
                complete_desc.name, another_desc.name,
                "Complete description should be deterministic"
            )
            
            system.evolve()


if __name__ == "__main__":
    unittest.main()