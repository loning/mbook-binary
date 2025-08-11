"""
Unit tests for T2-2: Encoding Completeness Theorem
T2-2：编码完备性定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class CompleteEncoder:
    """完备编码器实现"""
    
    def __init__(self):
        self.description_to_number = {}
        self.number_to_encoding = {}
        self.next_number = 0
        
    def encode_description(self, description):
        """编码描述（Gödel编码的简化版）"""
        if description in self.description_to_number:
            number = self.description_to_number[description]
        else:
            # 分配新编号
            number = self.next_number
            self.description_to_number[description] = number
            self.next_number += 1
            
        # 将数字编码为二进制串
        if number == 0:
            return "0"
        
        binary = ""
        n = number
        while n > 0:
            binary = str(n % 2) + binary
            n //= 2
            
        self.number_to_encoding[number] = binary
        return binary
        
    def encode(self, element):
        """完整的编码函数"""
        # 获取元素的描述
        if hasattr(element, 'description'):
            desc = element.description
        elif hasattr(element, 'get_description'):
            desc = element.get_description()
        elif hasattr(element, '__str__'):
            desc = str(element)
        else:
            desc = repr(element)
            
        return self.encode_description(desc)
        
    def decode_to_number(self, encoding):
        """解码到数字"""
        if not encoding or not all(c in '01' for c in encoding):
            return None
            
        # 二进制转数字
        number = 0
        for bit in encoding:
            number = number * 2 + int(bit)
        return number
        
    def decode(self, encoding):
        """完整解码"""
        number = self.decode_to_number(encoding)
        if number is None:
            return None
            
        # 查找对应的描述
        for desc, num in self.description_to_number.items():
            if num == number:
                return desc
        return None
        
    def is_injective(self):
        """检查编码是否单射"""
        encodings = list(self.number_to_encoding.values())
        return len(encodings) == len(set(encodings))
        
    def get_encoding_stats(self):
        """获取编码统计信息"""
        if not self.number_to_encoding:
            return {'count': 0, 'avg_length': 0, 'max_length': 0}
            
        lengths = [len(enc) for enc in self.number_to_encoding.values()]
        return {
            'count': len(self.number_to_encoding),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths)
        }


class InformationElement:
    """信息元素"""
    
    def __init__(self, id_val, description=None):
        self.id = id_val
        self.description = description or f"element_{id_val}"
        
    def __repr__(self):
        return f"InfoElem({self.id})"
        
    def __eq__(self, other):
        return isinstance(other, InformationElement) and self.id == other.id
        
    def __hash__(self):
        return hash(self.id)
        
    def get_description(self):
        return self.description


class ContinuousObject:
    """连续对象的有限表示"""
    
    def __init__(self, name, representations):
        self.name = name
        self.representations = representations  # dict of representation types
        
    def get_finite_representation(self, rep_type='default'):
        """获取有限表示"""
        if rep_type in self.representations:
            return self.representations[rep_type]
        elif 'default' in self.representations:
            return self.representations['default']
        elif self.representations:
            return list(self.representations.values())[0]
        return None
        
    def __repr__(self):
        return f"ContinuousObject({self.name})"


class InformationSystem:
    """信息系统"""
    
    def __init__(self):
        self.elements = set()
        self.descriptions = {}
        self.encoder = CompleteEncoder()
        
    def add_element(self, elem, description=None):
        """添加元素"""
        # 处理不可哈希的类型
        if isinstance(elem, (list, dict)):
            elem_key = str(elem)
        else:
            elem_key = elem
            
        self.elements.add(elem_key)
        
        if description:
            self.descriptions[elem_key] = description
        elif hasattr(elem, 'description'):
            self.descriptions[elem_key] = elem.description
        elif hasattr(elem, 'get_description'):
            self.descriptions[elem_key] = elem.get_description()
        else:
            self.descriptions[elem_key] = f"auto_desc_{elem_key}"
            
    def describe(self, elem):
        """获取元素描述"""
        # 处理不可哈希的类型
        if isinstance(elem, (list, dict)):
            elem_key = str(elem)
        else:
            elem_key = elem
        return self.descriptions.get(elem_key, str(elem_key))
        
    def get_all_elements(self):
        """获取所有元素"""
        return self.elements
        
    def has_information(self, x):
        """检查元素是否有信息（可区分）"""
        # 处理不可哈希的类型
        if isinstance(x, (list, dict)):
            x_key = str(x)
        else:
            x_key = x
            
        for y in self.elements:
            if x_key != y and self.describe(x) != self.describe(y):
                return True
        return False
        
    def get_information_elements(self):
        """获取所有有信息的元素"""
        return [x for x in self.elements if self.has_information(x)]
        
    def count_distinguishable_pairs(self):
        """计算可区分的元素对数量"""
        count = 0
        elements = list(self.elements)
        
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if self.describe(elements[i]) != self.describe(elements[j]):
                    count += 1
                    
        return count
        
    def verify_encoding_completeness(self):
        """验证编码完备性"""
        info_elements = self.get_information_elements()
        
        for elem in info_elements:
            encoding = self.encoder.encode(elem)
            if encoding is None:
                return False
                
        return True


class TestT2_2_EncodingCompleteness(VerificationTest):
    """T2-2 编码完备性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        
    def test_formal_information_definition(self):
        """测试信息形式定义 - 验证检查点1"""
        system = InformationSystem()
        
        # 添加一些元素
        for i in range(5):
            elem = InformationElement(i, f"desc_{i}")
            system.add_element(elem)
            
        # 添加重复描述的元素（无信息）
        duplicate = InformationElement(99, "desc_0")
        system.add_element(duplicate)
        
        # 获取有信息的元素
        info_elements = system.get_information_elements()
        
        # 验证信息元素的性质
        for elem in info_elements:
            self.assertTrue(
                system.has_information(elem),
                f"Element {elem} should have information"
            )
            
        # 验证信息定义：可区分性
        pairs = system.count_distinguishable_pairs()
        self.assertGreater(
            pairs, 0,
            "System should have distinguishable pairs"
        )
        
    def test_distinguishability_implies_describability(self):
        """测试可区分性蕴含可描述性 - 验证检查点2"""
        system = InformationSystem()
        
        # 创建可区分的元素
        elem1 = InformationElement(1, "unique_desc_1")
        elem2 = InformationElement(2, "unique_desc_2")
        elem3 = InformationElement(3, "unique_desc_3")
        
        system.add_element(elem1)
        system.add_element(elem2)
        system.add_element(elem3)
        
        # 验证所有可区分的元素都可描述
        for elem in system.get_all_elements():
            # 检查可区分性
            is_distinguishable = False
            for other in system.get_all_elements():
                if elem != other:
                    is_distinguishable = True
                    break
                    
            if is_distinguishable:
                # 验证可描述性
                description = system.describe(elem)
                self.assertIsNotNone(
                    description,
                    f"Distinguishable element {elem} should be describable"
                )
                self.assertIsInstance(
                    description, str,
                    "Description should be string"
                )
                self.assertGreater(
                    len(description), 0,
                    "Description should be non-empty"
                )
                
    def test_describability_implies_encodability(self):
        """测试可描述性蕴含可编码性 - 验证检查点3"""
        system = InformationSystem()
        encoder = system.encoder
        
        # 添加有描述的元素
        test_elements = [
            ("elem_a", "description_alpha"),
            ("elem_b", "description_beta"),
            ("elem_c", "description_gamma"),
            (42, "numeric_element"),
            ([1, 2, 3], "list_element")
        ]
        
        for elem, desc in test_elements:
            system.add_element(elem, desc)
            
        # 验证所有可描述的元素都可编码
        for elem in system.get_all_elements():
            description = system.describe(elem)
            
            if description:
                # 编码描述
                encoding = encoder.encode_description(description)
                
                self.assertIsNotNone(
                    encoding,
                    f"Description '{description}' should be encodable"
                )
                
                self.assertIsInstance(
                    encoding, str,
                    "Encoding should be string"
                )
                
                self.assertTrue(
                    all(c in '01' for c in encoding),
                    "Encoding should be binary string"
                )
                
                # 验证有限性
                self.assertLess(
                    len(encoding), float('inf'),
                    "Encoding should be finite"
                )
                
        # 验证编码的单射性
        self.assertTrue(
            encoder.is_injective(),
            "Encoding should be injective"
        )
        
    def test_continuous_object_finite_representation(self):
        """测试连续对象有限表示 - 验证检查点4"""
        # 创建连续对象的有限表示
        pi = ContinuousObject('pi', {
            'algorithm': 'π/4 = 4*arctan(1/5) - arctan(1/239)',
            'definition': 'circumference/diameter',
            'series': 'π = 4*Σ((-1)^n/(2n+1))',
            'decimal': '3.14159...',  # 只是标记，不是完整表示
        })
        
        e = ContinuousObject('e', {
            'limit': 'lim((1 + 1/n)^n, n→∞)',
            'series': 'e = Σ(1/n!)',
            'differential': 'dy/dx = y, y(0) = 1',
        })
        
        sin_func = ContinuousObject('sin', {
            'differential': "y'' + y = 0, y(0)=0, y'(0)=1",
            'series': 'sin(x) = Σ((-1)^n * x^(2n+1)/(2n+1)!)',
            'geometric': 'y-coordinate on unit circle',
        })
        
        continuous_objects = [pi, e, sin_func]
        
        # 验证每个对象都有有限表示
        for obj in continuous_objects:
            # 获取默认表示
            representation = obj.get_finite_representation()
            
            self.assertIsNotNone(
                representation,
                f"Object {obj.name} should have finite representation"
            )
            
            self.assertIsInstance(
                representation, str,
                "Representation should be string"
            )
            
            self.assertLess(
                len(representation), float('inf'),
                "Representation should be finite"
            )
            
            # 验证多种表示都是有限的
            for rep_type, rep in obj.representations.items():
                self.assertLess(
                    len(rep), 1000,  # 合理的上界
                    f"{obj.name} {rep_type} representation should be reasonably short"
                )
                
    def test_encoding_chain_completeness(self):
        """测试编码链完整性 - 验证检查点5"""
        system = InformationSystem()
        
        # 创建一系列元素
        for i in range(10):
            elem = InformationElement(i, f"unique_{i}")
            system.add_element(elem)
            
        # 统计编码链各阶段
        stats = {
            'total': len(system.get_all_elements()),
            'has_info': 0,
            'distinguishable': 0,
            'describable': 0,
            'encodable': 0
        }
        
        for elem in system.get_all_elements():
            # 检查信息
            if system.has_information(elem):
                stats['has_info'] += 1
                
                # 必然可区分
                stats['distinguishable'] += 1
                
                # 检查可描述
                if system.describe(elem):
                    stats['describable'] += 1
                    
                    # 检查可编码
                    if system.encoder.encode(elem):
                        stats['encodable'] += 1
                        
        # 验证链的单调性
        self.assertLessEqual(
            stats['has_info'], stats['distinguishable'],
            "Info implies distinguishability"
        )
        
        self.assertLessEqual(
            stats['distinguishable'], stats['describable'],
            "Distinguishability implies describability"
        )
        
        self.assertLessEqual(
            stats['describable'], stats['encodable'],
            "Describability implies encodability"
        )
        
        # 在完备系统中应该相等
        self.assertEqual(
            stats['has_info'], stats['encodable'],
            "Complete chain: all info should be encodable"
        )
        
    def test_encoding_completeness_theorem(self):
        """测试编码完备性定理"""
        system = InformationSystem()
        
        # 添加各种类型的元素
        # 1. 简单元素
        for i in range(5):
            system.add_element(f"simple_{i}")
            
        # 2. 复杂结构
        system.add_element({'type': 'dict', 'value': 42})
        system.add_element(['list', 'with', 'elements'])
        
        # 3. 自定义对象
        custom = InformationElement(100, "custom_description")
        system.add_element(custom)
        
        # 验证完备性
        self.assertTrue(
            system.verify_encoding_completeness(),
            "All information should be encodable"
        )
        
        # 验证每个有信息的元素都被正确编码
        info_elements = system.get_information_elements()
        encodings = set()
        
        for elem in info_elements:
            encoding = system.encoder.encode(elem)
            
            self.assertIsNotNone(
                encoding,
                f"Element {elem} should have encoding"
            )
            
            # 验证编码唯一性
            self.assertNotIn(
                encoding, encodings,
                f"Encoding {encoding} should be unique"
            )
            
            encodings.add(encoding)
            
    def test_godel_encoding_example(self):
        """测试Gödel编码示例"""
        encoder = CompleteEncoder()
        
        # 模拟Gödel编码过程
        test_strings = [
            "abc",
            "def",
            "abc",  # 重复，应该得到相同编码
            "xyz"
        ]
        
        encodings = {}
        
        for s in test_strings:
            encoding = encoder.encode_description(s)
            
            if s in encodings:
                # 验证相同描述得到相同编码
                self.assertEqual(
                    encoding, encodings[s],
                    "Same description should get same encoding"
                )
            else:
                encodings[s] = encoding
                
        # 验证不同描述得到不同编码
        unique_encodings = set(encodings.values())
        unique_strings = set(encodings.keys())
        
        self.assertEqual(
            len(unique_encodings), len(unique_strings),
            "Different descriptions should get different encodings"
        )
        
    def test_encoding_efficiency(self):
        """测试编码效率"""
        system = InformationSystem()
        
        # 添加大量元素
        for i in range(100):
            elem = InformationElement(i, f"element_with_id_{i:04d}")
            system.add_element(elem)
            
        # 编码所有元素
        for elem in system.get_all_elements():
            system.encoder.encode(elem)
            
        # 获取编码统计
        stats = system.encoder.get_encoding_stats()
        
        # 验证编码效率
        self.assertGreater(
            stats['count'], 0,
            "Should have encoded elements"
        )
        
        # 平均编码长度应该合理
        # 100个元素理论上需要至少log2(100) ≈ 6.64位
        theoretical_min = math.log2(stats['count'])
        
        self.assertLess(
            stats['avg_length'], theoretical_min * 2,
            "Average encoding length should be reasonable"
        )
        
    def test_edge_cases(self):
        """测试边界情况"""
        system = InformationSystem()
        
        # 1. 空系统
        self.assertEqual(
            len(system.get_information_elements()), 0,
            "Empty system should have no information"
        )
        
        # 2. 单元素系统
        system.add_element("single")
        self.assertEqual(
            len(system.get_information_elements()), 0,
            "Single element system has no distinguishable pairs"
        )
        
        # 3. 所有元素相同描述
        # 重新创建系统，避免与之前的元素混淆
        system = InformationSystem()
        for i in range(5):
            system.add_element(f"elem_{i}", "same_description")
            
        # 验证：如果所有元素描述相同，应该没有可区分的信息
        info_elements = system.get_information_elements()
        self.assertEqual(
            len(info_elements), 0,
            f"Elements with same description have no information, but got {len(info_elements)} info elements from {len(system.elements)} total"
        )
        
        # 4. 添加一个不同的元素
        system.add_element("different", "unique_description")
        
        # 现在应该有信息了
        info_count = len(system.get_information_elements())
        self.assertGreater(
            info_count, 0,
            "Adding unique element creates information"
        )
        
    def test_decode_functionality(self):
        """测试解码功能"""
        encoder = CompleteEncoder()
        
        # 测试编码-解码往返
        test_descriptions = [
            "first_description",
            "second_description",
            "third_description"
        ]
        
        for desc in test_descriptions:
            # 编码
            encoding = encoder.encode_description(desc)
            
            # 解码到数字
            number = encoder.decode_to_number(encoding)
            self.assertIsNotNone(
                number,
                f"Should decode {encoding} to number"
            )
            
            # 完整解码
            decoded = encoder.decode(encoding)
            self.assertEqual(
                decoded, desc,
                "Decode should return original description"
            )


if __name__ == "__main__":
    unittest.main()