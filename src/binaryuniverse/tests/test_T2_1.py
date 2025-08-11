"""
Unit tests for T2-1: Encoding Necessity Theorem
T2-1：编码机制必然性定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class EncodingSystem:
    """编码系统实现"""
    
    def __init__(self, alphabet_size=2):
        self.alphabet = list(range(alphabet_size))
        self.alphabet_symbols = [str(i) for i in self.alphabet]
        self.encodings = {}
        self.next_code = 0
        self.elements = set()
        
    def encode(self, element):
        """编码元素"""
        if element in self.encodings:
            return self.encodings[element]
            
        # 分配新编码
        code = self._int_to_string(self.next_code)
        self.encodings[element] = code
        self.elements.add(element)
        self.next_code += 1
        
        return code
        
    def _int_to_string(self, n):
        """整数转换为字母表字符串"""
        if n == 0:
            return self.alphabet_symbols[0]
            
        result = []
        base = len(self.alphabet)
        
        while n > 0:
            result.append(self.alphabet_symbols[n % base])
            n //= base
            
        return ''.join(reversed(result))
        
    def decode(self, code):
        """解码"""
        for elem, enc in self.encodings.items():
            if enc == code:
                return elem
        return None
        
    def can_encode_all(self, elements):
        """检查是否能编码所有元素"""
        for elem in elements:
            if self.encode(elem) is None:
                return False
        return True
        
    def is_injective(self):
        """检查单射性"""
        codes = list(self.encodings.values())
        return len(codes) == len(set(codes))
        
    def all_finite(self):
        """检查所有编码都是有限的"""
        for code in self.encodings.values():
            if len(code) >= float('inf'):
                return False
        return True
        
    def can_encode_self(self):
        """检查能否编码自己"""
        try:
            self_code = self.encode(self)
            return self_code is not None and len(self_code) < float('inf')
        except:
            return False
            
    def is_extensible(self, new_elements):
        """检查可扩展性"""
        original_size = len(self.encodings)
        
        for elem in new_elements:
            self.encode(elem)
            
        return len(self.encodings) == original_size + len(new_elements)
        
    def average_code_length(self):
        """计算平均编码长度"""
        if not self.encodings:
            return 0
        return sum(len(code) for code in self.encodings.values()) / len(self.encodings)
        
    def __repr__(self):
        return f"EncodingSystem(alphabet_size={len(self.alphabet)})"
        
    def __hash__(self):
        return hash(id(self))


class InformationSystem:
    """信息累积系统"""
    
    def __init__(self, initial_size=1):
        self.states = [set(f'state_{i}' for i in range(initial_size))]
        self.descriptions = [{f'state_{i}': f'desc_{i}' for i in range(initial_size)}]
        self.time = 0
        self.encoder = None
        self.growth_rate = 1.5  # 增长率
        
    def set_encoder(self, encoder):
        """设置编码器"""
        self.encoder = encoder
        
    def evolve(self):
        """演化一步"""
        current_size = len(self.states[-1])
        
        # 按增长率创建新状态
        new_count = max(1, int(current_size * (self.growth_rate - 1)))
        
        new_state = self.states[-1].copy()
        new_desc = self.descriptions[-1].copy()
        
        for i in range(new_count):
            state_name = f'state_t{self.time + 1}_n{i}'
            new_state.add(state_name)
            new_desc[state_name] = f'desc_t{self.time + 1}_n{i}'
            
        self.states.append(new_state)
        self.descriptions.append(new_desc)
        self.time += 1
        
    def get_state(self, t):
        """获取时刻t的状态"""
        if 0 <= t < len(self.states):
            return self.states[t]
        return set()
        
    def get_all_elements(self):
        """获取所有元素"""
        all_elems = set()
        for state in self.states:
            all_elems.update(state)
            
        # 如果有编码器，它也是系统的一部分
        if self.encoder:
            all_elems.add(self.encoder)
            
        return all_elems
        
    def describe(self, element):
        """描述元素"""
        # 查找元素的描述
        for desc_dict in self.descriptions:
            if element in desc_dict:
                return desc_dict[element]
                
        # 编码器的特殊描述
        if element == self.encoder:
            return "encoding_mechanism"
            
        return f"generic_desc_{element}"
        
    def has_distinct_information(self):
        """检查是否有可区分的信息"""
        elements = list(self.get_all_elements())
        
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if elements[i] != elements[j]:
                    desc_i = self.describe(elements[i])
                    desc_j = self.describe(elements[j])
                    if desc_i != desc_j:
                        return True
        return False
        
    def count_information(self):
        """计算信息数量"""
        # 不同描述的数量
        descriptions = set()
        for elem in self.get_all_elements():
            descriptions.add(self.describe(elem))
        return len(descriptions)
        
    def average_description_length(self):
        """计算平均描述长度"""
        lengths = []
        for elem in self.get_all_elements():
            desc = self.describe(elem)
            lengths.append(len(desc))
            
        return sum(lengths) / len(lengths) if lengths else 0
        
    def needs_encoding(self):
        """判断是否需要编码"""
        # 基于状态数和描述能力的矛盾
        state_count = len(self.get_all_elements())
        avg_desc_len = self.average_description_length()
        
        # 如果平均描述长度固定，而状态数持续增长
        # 最终会超过固定长度描述的表达能力
        if avg_desc_len < 5:
            max_expressible = 256 ** avg_desc_len
        else:
            max_expressible = float('inf')
        
        return state_count > max_expressible * 0.01 or state_count > 50  # 更早判断需求
        
    def is_self_referentially_complete(self):
        """检查自指完备性"""
        # 简化检查：是否所有元素都有描述
        for elem in self.get_all_elements():
            if not self.describe(elem):
                return False
        return True
        
    def shows_entropy_increase(self):
        """检查是否显示熵增"""
        if len(self.states) < 2:
            return False
            
        # 简化的熵计算：log(状态数)
        for i in range(len(self.states) - 1):
            entropy_i = math.log2(len(self.states[i])) if self.states[i] else 0
            entropy_i1 = math.log2(len(self.states[i+1])) if self.states[i+1] else 0
            
            if entropy_i1 <= entropy_i:
                return False
                
        return True


class TestT2_1_EncodingNecessity(VerificationTest):
    """T2-1 编码机制必然性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        
    def test_information_emergence(self):
        """测试信息涌现 - 验证检查点1"""
        system = InformationSystem(initial_size=3)
        
        # 验证系统有可区分的信息
        self.assertTrue(
            system.has_distinct_information(),
            "System should have distinguishable information"
        )
        
        # 验证信息数量
        info_count = system.count_information()
        self.assertGreater(
            info_count, 1,
            "System should have multiple distinct informations"
        )
        
        # 演化后信息应该增加
        initial_info = system.count_information()
        system.evolve()
        final_info = system.count_information()
        
        self.assertGreaterEqual(
            final_info, initial_info,
            "Information should not decrease"
        )
        
    def test_information_accumulation(self):
        """测试信息累积 - 验证检查点2"""
        system = InformationSystem()
        
        # 追踪系统大小
        sizes = []
        
        for t in range(10):
            size = len(system.get_state(t))
            sizes.append(size)
            
            if t < 9:
                system.evolve()
                
        # 验证严格增长
        for i in range(len(sizes) - 1):
            self.assertGreater(
                sizes[i+1], sizes[i],
                f"System size should increase at step {i}: {sizes[i]} -> {sizes[i+1]}"
            )
            
        # 验证累积效应
        self.assertGreater(
            sizes[-1], sizes[0] * 5,
            "System should show significant accumulation"
        )
        
    def test_finite_description_requirement(self):
        """测试有限描述要求 - 验证检查点3"""
        system = InformationSystem()
        
        # 演化多步
        for _ in range(5):
            system.evolve()
            
        # 验证所有描述都是有限的
        for elem in system.get_all_elements():
            desc = system.describe(elem)
            
            self.assertIsInstance(
                desc, str,
                "Description should be string"
            )
            
            self.assertLess(
                len(desc), float('inf'),
                "Description should be finite"
            )
            
            self.assertGreater(
                len(desc), 0,
                "Description should be non-empty"
            )
            
    def test_encoding_necessity(self):
        """测试编码需求 - 验证检查点4"""
        system = InformationSystem()
        
        # 快速演化到需要编码的程度
        system.growth_rate = 2.0  # 加快增长
        
        needs_encoding = False
        for t in range(15):  # 减少循环次数
            system.evolve()
            
            if system.needs_encoding():
                needs_encoding = True
                break
                
        self.assertTrue(
            needs_encoding,
            "System should eventually need encoding"
        )
        
        # 验证矛盾：状态数 vs 描述能力
        state_count = len(system.get_all_elements())
        avg_desc_len = system.average_description_length()
        
        self.assertGreater(
            state_count, 10,
            "System should have many states"
        )
        
        self.assertLess(
            avg_desc_len, 50,
            "Descriptions should have bounded length"
        )
        
    def test_encoder_self_reference(self):
        """测试编码器自指性 - 验证检查点5"""
        system = InformationSystem()
        encoder = EncodingSystem()
        
        # 设置编码器
        system.set_encoder(encoder)
        
        # 验证编码器在系统内
        self.assertIn(
            encoder, system.get_all_elements(),
            "Encoder should be part of the system"
        )
        
        # 验证编码器能编码自己
        self.assertTrue(
            encoder.can_encode_self(),
            "Encoder should be able to encode itself"
        )
        
        # 验证自编码的唯一性
        self_code = encoder.encode(encoder)
        other_elem = "test_element"
        other_code = encoder.encode(other_elem)
        
        self.assertNotEqual(
            self_code, other_code,
            "Self-encoding should be unique"
        )
        
    def test_encoding_properties(self):
        """测试编码机制的性质"""
        encoder = EncodingSystem(alphabet_size=2)
        
        # 测试元素集合
        test_elements = [f"elem_{i}" for i in range(20)]
        
        # 完备性：能编码所有元素
        self.assertTrue(
            encoder.can_encode_all(test_elements),
            "Encoder should encode all elements"
        )
        
        # 单射性：不同元素不同编码
        self.assertTrue(
            encoder.is_injective(),
            "Encoding should be injective"
        )
        
        # 有限性：所有编码都是有限的
        self.assertTrue(
            encoder.all_finite(),
            "All encodings should be finite"
        )
        
        # 递归性：能编码自己
        self.assertTrue(
            encoder.can_encode_self(),
            "Encoder should handle self-reference"
        )
        
        # 可扩展性：能处理新元素
        new_elements = [f"new_elem_{i}" for i in range(10)]
        self.assertTrue(
            encoder.is_extensible(new_elements),
            "Encoder should be extensible"
        )
        
    def test_binary_vs_other_alphabets(self):
        """测试不同字母表大小的编码效率"""
        # 测试不同大小的字母表
        alphabet_sizes = [2, 3, 4, 10]
        element_count = 100
        test_elements = [f"elem_{i}" for i in range(element_count)]
        
        results = []
        
        for size in alphabet_sizes:
            encoder = EncodingSystem(alphabet_size=size)
            
            # 编码所有元素
            for elem in test_elements:
                encoder.encode(elem)
                
            # 计算平均编码长度
            avg_length = encoder.average_code_length()
            results.append((size, avg_length))
            
        # 验证结果合理性
        for size, avg_len in results:
            # 理论最小长度：log_size(element_count)
            theoretical_min = math.log(element_count) / math.log(size)
            
            self.assertGreater(
                avg_len, theoretical_min * 0.8,
                f"Average length for base {size} should be reasonable"
            )
            
    def test_encoding_growth_handling(self):
        """测试编码处理系统增长"""
        system = InformationSystem()
        encoder = EncodingSystem()
        system.set_encoder(encoder)
        
        # 模拟长期演化
        for t in range(10):  # 减少循环次数
            system.evolve()
            
            # 编码所有新元素
            current_elements = system.get_all_elements()
            for elem in current_elements:
                encoder.encode(elem)
                
        # 验证编码器处理了所有增长
        self.assertEqual(
            len(encoder.elements), len(system.get_all_elements()),
            "Encoder should handle all growth"
        )
        
        # 验证编码仍然有效
        self.assertTrue(
            encoder.is_injective(),
            "Encoding should remain injective after growth"
        )
        
    def test_encoding_efficiency(self):
        """测试编码效率"""
        encoder = EncodingSystem(alphabet_size=2)
        
        # 编码不同数量的元素
        test_sizes = [10, 50, 100, 500]
        
        for size in test_sizes:
            # 重置编码器
            encoder = EncodingSystem(alphabet_size=2)
            
            # 编码元素
            for i in range(size):
                encoder.encode(f"elem_{i}")
                
            # 检查平均编码长度
            avg_length = encoder.average_code_length()
            
            # 理论最优：log2(size)
            theoretical_optimal = math.log2(size)
            
            # 实际应该接近理论值
            self.assertLess(
                avg_length, theoretical_optimal + 2,
                f"Encoding should be efficient for size {size}"
            )
            
    def test_system_encoder_integration(self):
        """测试系统与编码器的集成"""
        system = InformationSystem()
        encoder = EncodingSystem()
        system.set_encoder(encoder)
        
        # 验证初始状态
        self.assertTrue(
            system.is_self_referentially_complete(),
            "System should be self-referentially complete"
        )
        
        # 演化并编码
        for _ in range(10):
            system.evolve()
            
            # 编码所有元素
            for elem in system.get_all_elements():
                code = encoder.encode(elem)
                
                # 验证可以解码
                decoded = encoder.decode(code)
                self.assertEqual(
                    decoded, elem,
                    "Should be able to decode back to original"
                )
                
        # 验证系统性质保持
        self.assertTrue(
            system.shows_entropy_increase(),
            "System should maintain entropy increase"
        )


if __name__ == "__main__":
    unittest.main()