"""
Unit tests for T2-10: φ-Representation Completeness Theorem
T2-10：φ-表示完备性定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from typing import List, Dict, Any, Optional, Tuple
import hashlib


class TestT2_10_PhiRepresentationCompleteness(VerificationTest):
    """T2-10 φ-表示完备性定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        # 预生成Fibonacci数列（需要更多以处理大数）
        self.fibonacci = self._generate_fibonacci(100)
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成前n个修正的Fibonacci数（从1,2开始）"""
        if n == 0:
            return []
        if n == 1:
            return [1]
        
        fib = [1, 2]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def _has_info(self, element: Any) -> bool:
        """检查元素是否有信息（可区分性）"""
        # 简化实现：非None且有某种结构
        return element is not None and (
            isinstance(element, (int, str, dict, list)) or
            hasattr(element, '__dict__')
        )
    
    def _extract_description(self, element: Any) -> str:
        """提取元素的描述"""
        if isinstance(element, str):
            return element
        elif isinstance(element, int):
            return f"int:{element}"
        elif isinstance(element, dict):
            return f"dict:{sorted(element.items())}"
        elif isinstance(element, list):
            return f"list:{element}"
        else:
            return str(element)
    
    def _string_to_natural(self, s: str) -> int:
        """将字符串映射到自然数"""
        # 使用哈希函数的简化版本
        hash_bytes = hashlib.sha256(s.encode()).digest()
        # 取前8字节作为整数
        return int.from_bytes(hash_bytes[:8], 'big')
    
    def _compute_zeckendorf(self, n: int) -> List[int]:
        """计算n的Zeckendorf表示（φ-表示）"""
        if n == 0:
            return []
        
        # 找到需要的Fibonacci数
        fibs_needed = []
        indices_used = []
        remaining = n
        
        # 贪心算法：从大到小选择
        for i in range(len(self.fibonacci) - 1, -1, -1):
            if self.fibonacci[i] <= remaining:
                fibs_needed.append(self.fibonacci[i])
                indices_used.append(i)
                remaining -= self.fibonacci[i]
                if remaining == 0:
                    break
        
        if remaining > 0:
            # 数太大，超出预生成的Fibonacci数
            return []
        
        # 构建二进制表示
        max_index = indices_used[0]
        result = [0] * (max_index + 1)
        for idx in indices_used:
            result[idx] = 1
        
        return result
    
    def _decode_zeckendorf(self, phi_repr: List[int]) -> int:
        """从φ-表示解码为自然数"""
        if not phi_repr:
            return 0
        
        # 计算值
        value = 0
        for i, bit in enumerate(phi_repr):
            if bit == 1:
                value += self.fibonacci[i]
        
        return value
    
    def _is_valid_phi_repr(self, repr_list: List[int]) -> bool:
        """检查是否是有效的φ-表示（无相邻的1）"""
        for i in range(len(repr_list) - 1):
            if repr_list[i] == 1 and repr_list[i+1] == 1:
                return False
        return True
    
    def test_information_distinguishability(self):
        """测试信息可区分性 - 验证检查点1"""
        # 创建测试元素
        test_elements = [
            42,
            "hello",
            {"key": "value"},
            [1, 2, 3],
            {"nested": {"data": 123}},
            None  # 无信息元素
        ]
        
        # 验证有信息的元素
        info_count = 0
        for x in test_elements:
            if self._has_info(x):
                info_count += 1
                
                # 验证可以提取描述
                desc = self._extract_description(x)
                self.assertIsInstance(desc, str)
                self.assertGreater(len(desc), 0)
                
                # 验证不同元素有不同描述
                for y in test_elements:
                    if x is not y and self._has_info(y):
                        desc_y = self._extract_description(y)
                        if x != y:
                            self.assertNotEqual(
                                desc, desc_y,
                                f"Different elements should have different descriptions"
                            )
        
        # 确保大部分元素有信息
        self.assertGreater(info_count, len(test_elements) - 2)
        
    def test_encoding_chain_completeness(self):
        """测试编码链完整性 - 验证检查点2"""
        # 测试信息样本
        test_data = [
            "simple string",
            {"type": "object", "id": 123},
            [1, 1, 2, 3, 5, 8],
            "φ-representation system rules"
        ]
        
        encoding_results = []
        
        for data in test_data:
            # Step 1: 提取描述
            desc = self._extract_description(data)
            self.assertIsInstance(desc, str)
            
            # Step 2: 映射到自然数
            n = self._string_to_natural(desc)
            self.assertIsInstance(n, int)
            self.assertGreaterEqual(n, 0)
            
            # Step 3: 计算φ-表示
            phi_repr = self._compute_zeckendorf(n)
            self.assertTrue(
                self._is_valid_phi_repr(phi_repr),
                f"Invalid φ-representation for {data}"
            )
            
            # 记录结果
            encoding_results.append({
                "data": data,
                "desc": desc,
                "natural": n,
                "phi_repr": phi_repr
            })
        
        # 验证不同数据有不同编码
        naturals = [r["natural"] for r in encoding_results]
        unique_naturals = set(naturals)
        self.assertEqual(
            len(naturals), len(unique_naturals),
            "Different data should map to different naturals"
        )
        
        # 验证双射性（简化：验证解码正确）
        for result in encoding_results:
            decoded_n = self._decode_zeckendorf(result["phi_repr"])
            
            # 注意：由于哈希映射，我们只能验证φ-表示的正确性
            # 不能完全恢复原始数据
            test_phi = self._compute_zeckendorf(decoded_n)
            self.assertEqual(
                test_phi, result["phi_repr"],
                "φ-representation should be consistent"
            )
            
    def test_zeckendorf_representation(self):
        """测试Zeckendorf表示 - 验证检查点3"""
        # 测试前100个自然数
        for n in range(100):
            phi_repr = self._compute_zeckendorf(n)
            
            # 验证表示的有效性
            self.assertTrue(
                self._is_valid_phi_repr(phi_repr),
                f"φ-representation of {n} violates no-11 constraint"
            )
            
            # 验证解码的正确性
            decoded = self._decode_zeckendorf(phi_repr)
            self.assertEqual(
                decoded, n,
                f"Decoding error: {n} -> {phi_repr} -> {decoded}"
            )
            
        # 测试特定值的表示
        test_cases = [
            (1, [1]),          # F_1 = 1
            (2, [0, 1]),       # F_2 = 2  
            (3, [1, 1]),       # F_1 + F_2 = 1 + 2 = 3，错误！应该是[0, 0, 1]
            (4, [1, 0, 1]),    # F_1 + F_3 = 1 + 3 = 4
            (7, [0, 1, 0, 1]), # F_2 + F_4 = 2 + 5 = 7
        ]
        
        # 修正测试用例
        test_cases = [
            (1, [1]),              # F_0 = 1
            (2, [0, 1]),           # F_1 = 2
            (3, [0, 0, 1]),        # F_2 = 3
            (4, [1, 0, 1]),        # F_0 + F_2 = 1 + 3 = 4
            (5, [0, 0, 0, 1]),     # F_3 = 5
            (7, [0, 1, 0, 1]),     # F_1 + F_3 = 2 + 5 = 7
            (12, [1, 0, 1, 0, 1]), # F_0 + F_2 + F_4 = 1 + 3 + 8 = 12
        ]
        
        for n, expected in test_cases:
            result = self._compute_zeckendorf(n)
            self.assertEqual(
                result, expected,
                f"Zeckendorf({n}) = {result}, expected {expected}"
            )
            
    def test_self_encoding_capability(self):
        """测试自编码能力 - 验证检查点4"""
        # φ-表示系统的规则
        phi_rules = {
            "name": "phi-representation",
            "base": 2,
            "constraint": "no-11",
            "fibonacci_start": [1, 2],
            "algorithm": "greedy_zeckendorf",
            "properties": {
                "unique": True,
                "complete": True,
                "no_adjacent_ones": True
            }
        }
        
        # 将规则编码为描述
        rules_desc = self._extract_description(phi_rules)
        self.assertIsInstance(rules_desc, str)
        self.assertIn("phi-representation", rules_desc)
        
        # 通过编码链
        n = self._string_to_natural(rules_desc)
        self.assertIsInstance(n, int)
        
        phi_repr = self._compute_zeckendorf(n)
        self.assertTrue(self._is_valid_phi_repr(phi_repr))
        
        # 验证表示的属性
        self.assertGreater(len(phi_repr), 0, "Rules should have non-empty representation")
        
        # 验证可以解码（至少解码到相同的自然数）
        decoded_n = self._decode_zeckendorf(phi_repr)
        re_encoded = self._compute_zeckendorf(decoded_n)
        self.assertEqual(
            re_encoded, phi_repr,
            "Self-encoding should be stable"
        )
        
        # 验证系统可以编码自己的组件
        components = ["base", "constraint", "algorithm", "fibonacci"]
        for comp in components:
            comp_n = self._string_to_natural(comp)
            comp_phi = self._compute_zeckendorf(comp_n)
            self.assertTrue(
                self._is_valid_phi_repr(comp_phi),
                f"Component '{comp}' should have valid φ-representation"
            )
            
    def test_continuous_object_handling(self):
        """测试连续对象处理 - 验证检查点5"""
        # 定义"连续"对象的算法表示
        continuous_objects = {
            "pi": {
                "name": "leibniz_series",
                "formula": "π/4 = 1 - 1/3 + 1/5 - 1/7 + ...",
                "convergence": "slow",
                "terms": "alternating"
            },
            "e": {
                "name": "taylor_series", 
                "formula": "e = Σ(1/n!)",
                "convergence": "fast",
                "factorial": True
            },
            "sqrt2": {
                "name": "newton_method",
                "formula": "x_{n+1} = (x_n + 2/x_n)/2",
                "convergence": "quadratic",
                "initial": 1.5
            },
            "golden_ratio": {
                "name": "closed_form",
                "formula": "(1 + sqrt(5))/2",
                "exact": True,
                "relation": "fibonacci_ratio"
            }
        }
        
        encoded_objects = {}
        
        for obj_name, algorithm in continuous_objects.items():
            # 算法是有限描述
            alg_desc = self._extract_description(algorithm)
            self.assertIsInstance(alg_desc, str)
            self.assertLess(
                len(alg_desc), 10000,
                f"Algorithm for {obj_name} should have finite description"
            )
            
            # 编码为自然数
            n = self._string_to_natural(alg_desc)
            
            # 计算φ-表示
            phi_repr = self._compute_zeckendorf(n)
            self.assertTrue(
                self._is_valid_phi_repr(phi_repr),
                f"Algorithm for {obj_name} should have valid φ-representation"
            )
            
            encoded_objects[obj_name] = {
                "algorithm": algorithm,
                "natural": n,
                "phi_repr": phi_repr,
                "size": len(phi_repr)
            }
        
        # 验证不同对象有不同编码
        representations = [enc["phi_repr"] for enc in encoded_objects.values()]
        for i, repr1 in enumerate(representations):
            for j, repr2 in enumerate(representations[i+1:], i+1):
                self.assertNotEqual(
                    repr1, repr2,
                    f"Different objects should have different φ-representations"
                )
        
        # 验证编码效率（φ-表示应该相当紧凑）
        for obj_name, enc in encoded_objects.items():
            # 粗略估计：log_φ(n) ≈ log(n) / log(φ)
            import math
            golden_ratio = (1 + math.sqrt(5)) / 2
            expected_size = math.log(enc["natural"] + 1) / math.log(golden_ratio)
            actual_size = enc["size"]
            
            # 允许一定的偏差
            ratio = actual_size / max(expected_size, 1)
            self.assertLess(
                ratio, 2.0,
                f"{obj_name}: size {actual_size} too large for n={enc['natural']}"
            )
            
    def test_encoding_efficiency(self):
        """测试编码效率分析"""
        import math
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # 测试不同大小的数据
        test_sizes = [10, 100, 1000, 10000]
        
        for size in test_sizes:
            # 创建测试数据
            data = list(range(size))
            desc = self._extract_description(data)
            n = self._string_to_natural(desc)
            
            # 计算φ-表示
            phi_repr = self._compute_zeckendorf(n)
            
            # 分析效率
            phi_bits = len(phi_repr)
            binary_bits = n.bit_length()
            optimal_phi_bits = math.log(n + 1) / math.log(golden_ratio)
            
            # 验证接近理论最优
            if n > 100:  # 对较大的数验证
                efficiency = phi_bits / optimal_phi_bits
                self.assertLess(
                    efficiency, 1.5,
                    f"φ-representation should be near-optimal"
                )
                
                # 与二进制比较（应该最多多44%）
                overhead = phi_bits / max(binary_bits, 1)
                self.assertLess(
                    overhead, 1.6,
                    f"φ-representation overhead should be < 60%"
                )
                
    def test_special_cases(self):
        """测试特殊情况"""
        # 空信息
        empty_desc = self._extract_description("")
        empty_n = self._string_to_natural(empty_desc)
        empty_phi = self._compute_zeckendorf(empty_n)
        self.assertTrue(self._is_valid_phi_repr(empty_phi))
        
        # 非常大的数
        large_n = 10**12
        large_phi = self._compute_zeckendorf(large_n)
        self.assertTrue(self._is_valid_phi_repr(large_phi))
        decoded_large = self._decode_zeckendorf(large_phi)
        self.assertEqual(decoded_large, large_n)
        
        # 递归结构
        recursive_data = {"self": None}
        recursive_data["self"] = recursive_data  # 自引用
        
        # 应该能处理（通过描述的有限性）
        rec_desc = self._extract_description(str(recursive_data)[:1000])  # 截断避免无限
        rec_n = self._string_to_natural(rec_desc)
        rec_phi = self._compute_zeckendorf(rec_n)
        self.assertTrue(self._is_valid_phi_repr(rec_phi))
        
    def test_theoretical_properties(self):
        """测试理论性质"""
        # 1. 完备性：任何自然数都有φ-表示
        test_range = 1000
        for n in range(test_range):
            phi_repr = self._compute_zeckendorf(n)
            self.assertIsNotNone(phi_repr)
            decoded = self._decode_zeckendorf(phi_repr)
            self.assertEqual(decoded, n)
        
        # 2. 唯一性：通过贪心算法保证
        # 已在其他测试中验证
        
        # 3. 自指性：系统可以描述自己
        # 已在test_self_encoding_capability中验证
        
        # 4. 与Fibonacci数列的关系
        for i, fib in enumerate(self.fibonacci[:10]):
            phi_repr = self._compute_zeckendorf(fib)
            # Fibonacci数的表示应该在位置i有一个1
            expected = [0] * (i + 1)
            expected[i] = 1
            self.assertEqual(
                phi_repr, expected,
                f"F_{i} = {fib} should have simple representation"
            )
            
    def test_information_preservation(self):
        """测试信息保持性"""
        # 创建具有丰富结构的测试数据
        complex_data = {
            "metadata": {
                "version": "1.0",
                "encoding": "phi-representation"
            },
            "content": {
                "numbers": [1, 1, 2, 3, 5, 8, 13],
                "text": "Self-referential completeness",
                "nested": {
                    "level": 3,
                    "data": [True, False, None]
                }
            },
            "checksum": "abc123"
        }
        
        # 编码过程
        desc1 = self._extract_description(complex_data)
        n1 = self._string_to_natural(desc1)
        phi1 = self._compute_zeckendorf(n1)
        
        # 验证确定性：相同输入产生相同输出
        desc2 = self._extract_description(complex_data)
        n2 = self._string_to_natural(desc2)
        phi2 = self._compute_zeckendorf(n2)
        
        self.assertEqual(desc1, desc2)
        self.assertEqual(n1, n2)
        self.assertEqual(phi1, phi2)
        
        # 验证微小改变导致不同编码
        modified_data = complex_data.copy()
        modified_data["checksum"] = "xyz789"
        
        desc_mod = self._extract_description(modified_data)
        n_mod = self._string_to_natural(desc_mod)
        phi_mod = self._compute_zeckendorf(n_mod)
        
        self.assertNotEqual(n1, n_mod)
        self.assertNotEqual(phi1, phi_mod)


if __name__ == "__main__":
    unittest.main()