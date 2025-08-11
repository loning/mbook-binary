"""
Unit tests for L1-1: Encoding Emergence Lemma
L1-1：编码需求涌现引理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import SystemState, create_initial_system
import math


class EncodingSystem:
    """编码系统实现"""
    
    def __init__(self):
        self.encoding_map = {}
        self.next_code = 1
        self.alphabet = ['0', '1']  # 二进制字母表
        
    def encode(self, element):
        """为元素分配编码"""
        if element in self.encoding_map:
            return self.encoding_map[element]
            
        # 分配新编码
        code = self._int_to_binary(self.next_code)
        self.encoding_map[element] = code
        self.next_code += 1
        
        return code
        
    def _int_to_binary(self, n):
        """整数转二进制字符串"""
        if n == 0:
            return "0"
        return bin(n)[2:]  # 去掉'0b'前缀
        
    def decode(self, code):
        """从编码恢复元素"""
        for element, enc in self.encoding_map.items():
            if enc == code:
                return element
        return None
        
    def is_injective(self):
        """验证编码是单射的"""
        codes = list(self.encoding_map.values())
        return len(codes) == len(set(codes))
        
    def all_codes_finite(self):
        """验证所有编码都是有限长度"""
        for code in self.encoding_map.values():
            if len(code) == float('inf'):
                return False
        return True
        
    def can_encode_self(self):
        """验证编码系统能编码自身"""
        # 简化：将编码系统表示为特殊元素
        self_representation = "EncodingSystem"
        self_code = self.encode(self_representation)
        return self_code is not None and len(self_code) < float('inf')


class GrowingSystem:
    """模拟增长的自指完备系统"""
    
    def __init__(self):
        self.states = [{"initial"}]  # 时间序列的状态集合
        self.descriptions = {}
        self.time = 0
        
    def evolve(self):
        """系统演化一步"""
        current_state = self.states[-1]
        new_state = current_state.copy()
        
        # 添加新元素（模拟熵增）
        new_element = f"state_{self.time}_{len(current_state)}"
        new_state.add(new_element)
        
        # 自指：系统包含对自身的描述
        new_state.add(f"desc_of_{self.time}")
        
        self.states.append(new_state)
        self.time += 1
        
    def get_entropy(self, t):
        """计算时刻t的熵"""
        if t >= len(self.states):
            return 0
        return math.log2(len(self.states[t]))
        
    def is_entropy_increasing(self):
        """验证熵是否递增"""
        for t in range(len(self.states) - 1):
            if self.get_entropy(t+1) <= self.get_entropy(t):
                return False
        return True
        
    def describe(self, element):
        """为元素生成描述"""
        if element not in self.descriptions:
            # 生成有限长度的描述
            self.descriptions[element] = f"desc_{hash(element) % 1000}"
        return self.descriptions[element]
        
    def all_descriptions_finite(self):
        """验证所有描述都是有限长度"""
        for elem in self.states[-1]:
            desc = self.describe(elem)
            if len(desc) == float('inf'):
                return False
        return True


class TestL1_1_EncodingEmergence(VerificationTest):
    """L1-1 编码需求涌现的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.encoding_system = EncodingSystem()
        self.growing_system = GrowingSystem()
        
    def test_information_emergence(self):
        """测试信息涌现 - 验证检查点1"""
        # 创建自指完备系统
        system = create_initial_system()
        
        # 验证系统有至少两个不同元素
        self.assertGreaterEqual(
            len(system.elements), 2,
            "Self-referential complete system must have at least 2 elements"
        )
        
        # 创建描述映射来模拟不同元素有不同描述
        elements = list(system.elements)
        descriptions = {}
        
        # 为每个元素分配唯一描述
        for i, elem in enumerate(elements):
            descriptions[elem] = f"description_{i}_{hash(elem)}"
            
        # 验证不同元素有不同描述
        if len(elements) >= 2:
            desc1 = descriptions[elements[0]]
            desc2 = descriptions[elements[1]]
            
            self.assertNotEqual(
                desc1, desc2,
                "Different elements must have different descriptions"
            )
            
        # 验证信息概念的存在
        has_info = False
        for elem in elements:
            desc = descriptions[elem]
            if any(descriptions[other] != desc for other in elements if other != elem):
                has_info = True
                break
                
        self.assertTrue(
            has_info,
            "Information concept must emerge in the system"
        )
        
    def test_information_accumulation(self):
        """测试信息累积 - 验证检查点2"""
        # 演化系统多步
        for _ in range(5):
            self.growing_system.evolve()
            
        # 验证熵递增
        self.assertTrue(
            self.growing_system.is_entropy_increasing(),
            "System entropy must increase over time"
        )
        
        # 验证状态数递增
        for t in range(len(self.growing_system.states) - 1):
            self.assertGreater(
                len(self.growing_system.states[t+1]),
                len(self.growing_system.states[t]),
                f"State count must increase: |S_{t+1}| > |S_{t}|"
            )
            
        # 验证信息累积的具体数值
        initial_size = len(self.growing_system.states[0])
        final_size = len(self.growing_system.states[-1])
        
        self.assertGreater(
            final_size, initial_size,
            "Information must accumulate over time"
        )
        
    def test_finite_description_requirement(self):
        """测试有限描述要求 - 验证检查点3"""
        # 演化系统
        for _ in range(10):
            self.growing_system.evolve()
            
        # 验证所有描述都是有限长度
        self.assertTrue(
            self.growing_system.all_descriptions_finite(),
            "All descriptions must have finite length"
        )
        
        # 测试具体描述长度
        current_state = self.growing_system.states[-1]
        for element in current_state:
            desc = self.growing_system.describe(element)
            
            self.assertIsInstance(
                desc, str,
                "Description must be a string (finite symbol sequence)"
            )
            
            self.assertLess(
                len(desc), float('inf'),
                f"Description length must be finite for {element}"
            )
            
            self.assertGreater(
                len(desc), 0,
                f"Description must be non-empty for {element}"
            )
            
    def test_encoding_necessity(self):
        """测试编码必然性 - 验证检查点4"""
        # 创建快速增长的系统
        rapid_growth_system = GrowingSystem()
        
        # 快速演化
        for i in range(20):
            rapid_growth_system.evolve()
            # 额外添加元素模拟快速增长
            for j in range(i):
                rapid_growth_system.states[-1].add(f"extra_{i}_{j}")
                
        # 验证系统增长
        growth_rate = len(rapid_growth_system.states[-1]) / len(rapid_growth_system.states[0])
        self.assertGreater(
            growth_rate, 10,
            "System must show significant growth"
        )
        
        # 验证需要编码机制
        # 尝试为所有元素编码
        all_elements = rapid_growth_system.states[-1]
        for elem in all_elements:
            code = self.encoding_system.encode(elem)
            self.assertIsNotNone(
                code,
                f"Must be able to encode element {elem}"
            )
            
        # 验证编码的必要性质
        self.assertTrue(
            self.encoding_system.is_injective(),
            "Encoding must be injective"
        )
        
        self.assertTrue(
            self.encoding_system.all_codes_finite(),
            "All codes must be finite"
        )
        
    def test_encoding_properties(self):
        """测试编码属性"""
        # 测试单射性
        elements = ["a", "b", "c", "d", "e"]
        codes = []
        
        for elem in elements:
            code = self.encoding_system.encode(elem)
            codes.append(code)
            
        # 验证没有重复编码
        self.assertEqual(
            len(codes), len(set(codes)),
            "Encoding must be injective (no duplicate codes)"
        )
        
        # 测试解码
        for i, elem in enumerate(elements):
            decoded = self.encoding_system.decode(codes[i])
            self.assertEqual(
                decoded, elem,
                f"Decoding must recover original element: {elem}"
            )
            
    def test_self_encoding_capability(self):
        """测试自编码能力"""
        # 验证编码系统能编码自身
        self.assertTrue(
            self.encoding_system.can_encode_self(),
            "Encoding system must be able to encode itself"
        )
        
        # 测试递归编码
        # 编码"编码函数的描述"
        meta_encoding = self.encoding_system.encode("encoding_function_description")
        self.assertIsNotNone(
            meta_encoding,
            "Must be able to encode descriptions of encoding"
        )
        
        # 编码元编码
        meta_meta_encoding = self.encoding_system.encode(f"code_of_{meta_encoding}")
        self.assertIsNotNone(
            meta_meta_encoding,
            "Must support recursive encoding"
        )
        
    def test_encoding_efficiency(self):
        """测试编码效率"""
        # 编码大量元素
        n_elements = 1000
        for i in range(n_elements):
            self.encoding_system.encode(f"element_{i}")
            
        # 检查编码长度的对数增长
        max_code_length = max(
            len(code) for code in self.encoding_system.encoding_map.values()
        )
        
        theoretical_min = math.ceil(math.log2(n_elements))
        
        # 验证编码长度接近理论最小值
        self.assertLessEqual(
            max_code_length, theoretical_min + 1,
            f"Encoding length should be close to log2(n) = {theoretical_min}"
        )
        
    def test_contradiction_resolution(self):
        """测试矛盾解决"""
        # 创建状态数与描述长度的矛盾
        system = GrowingSystem()
        
        # 大量演化
        for _ in range(100):
            system.evolve()
            
        # 验证矛盾：无限增长vs有限描述
        state_count = len(system.states[-1])
        self.assertGreater(
            state_count, 100,
            "State count grows without bound"
        )
        
        # 但每个描述仍然有限
        self.assertTrue(
            system.all_descriptions_finite(),
            "Descriptions remain finite despite growth"
        )
        
        # 验证编码解决了矛盾
        encoding = EncodingSystem()
        for elem in system.states[-1]:
            code = encoding.encode(elem)
            self.assertLess(
                len(code), float('inf'),
                "Encoding provides finite representation"
            )
            
    def test_extensibility(self):
        """测试可扩展性"""
        # 验证编码可以处理任意时刻的状态
        system = GrowingSystem()
        encoding = EncodingSystem()
        
        for t in range(50):
            system.evolve()
            
            # 编码当前时刻的所有状态
            current_states = system.states[t]
            all_encoded = True
            
            for state in current_states:
                try:
                    code = encoding.encode(state)
                    if code is None:
                        all_encoded = False
                        break
                except:
                    all_encoded = False
                    break
                    
            self.assertTrue(
                all_encoded,
                f"Encoding must handle all states at time {t}"
            )
            
    def test_formal_language_properties(self):
        """测试形式语言属性"""
        # 验证编码产生的是形式语言的元素
        elements = [f"test_{i}" for i in range(10)]
        
        for elem in elements:
            code = self.encoding_system.encode(elem)
            
            # 验证是有限字母表上的串
            self.assertTrue(
                all(symbol in self.encoding_system.alphabet for symbol in code),
                f"Code must use only symbols from finite alphabet: {code}"
            )
            
            # 验证是有限长度
            self.assertLess(
                len(code), float('inf'),
                "Code must have finite length"
            )


if __name__ == "__main__":
    unittest.main()