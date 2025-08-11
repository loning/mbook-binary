"""
Unit tests for D1-2: Binary Representation
D1-2：二进制表示的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import BinaryEncoding


class TestD1_2_BinaryRepresentation(VerificationTest):
    """D1-2二进制表示的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.encoder = BinaryEncoding(no_11_constraint=False)
        
    def test_encoding_injectivity(self):
        """测试编码单射性（唯一性）"""
        # 测试不同输入产生不同编码
        test_values = [0, 1, 2, 3, 10, 100, 255]
        encodings = {}
        
        for val in test_values:
            encoding = self.encoder.encode(val)
            # 验证编码唯一性
            self.assertNotIn(
                encoding,
                encodings.values(),
                f"Encoding for {val} should be unique"
            )
            encodings[val] = encoding
            
        # 验证解码能恢复原值
        for val, encoding in encodings.items():
            decoded = self.encoder.decode(encoding)
            self.assertEqual(
                decoded, val,
                f"Decoding should recover original value {val}"
            )
            
    def test_prefix_freedom(self):
        """测试前缀自由性"""
        # 生成一组编码
        values = range(10)
        encodings = [self.encoder.encode(v) for v in values]
        
        # 检查任何编码都不是其他编码的前缀
        for i, enc1 in enumerate(encodings):
            for j, enc2 in enumerate(encodings):
                if i != j:
                    self.assertFalse(
                        enc2.startswith(enc1),
                        f"Encoding {enc1} should not be prefix of {enc2}"
                    )
                    
    def test_self_embedding(self):
        """测试自嵌入性"""
        # 创建能编码自身的系统
        class SelfEncodingSystem:
            def __init__(self):
                self.encoder = BinaryEncoding()
                
            def encode_self(self):
                # 模拟编码函数编码自身
                # 使用函数的字符串表示作为简化
                self_description = "BinaryEncoding"
                return self.encoder.encode(self_description)
                
        system = SelfEncodingSystem()
        self_encoding = system.encode_self()
        
        # 验证能够编码自身
        self.assertIsNotNone(self_encoding)
        self.assertIsInstance(self_encoding, str)
        self.assertTrue(all(c in '01' for c in self_encoding))
        
    def test_encoding_closure(self):
        """测试编码封闭性"""
        test_inputs = [42, "hello", "世界"]
        
        for input_val in test_inputs:
            encoding = self.encoder.encode(input_val)
            
            # 验证编码结果是二进制字符串
            self.assertIsInstance(encoding, str)
            self.assertTrue(
                all(c in '01' for c in encoding),
                f"Encoding should only contain 0 and 1, got: {encoding}"
            )
            
    def test_binary_string_operations(self):
        """测试二进制字符串操作"""
        # 测试空串
        empty = ""
        self.assertEqual(len(empty), 0)
        
        # 测试连接
        s1 = "101"
        s2 = "110"
        concatenated = s1 + s2
        self.assertEqual(concatenated, "101110")
        
        # 测试前缀判断
        self.assertTrue(s1 == concatenated[:len(s1)])
        self.assertFalse(s2 == concatenated[:len(s2)])
        
    def test_encoding_length_bounds(self):
        """测试编码长度界限"""
        # 对于n个不同的值，编码长度至少需要log2(n)位
        n_values = 16
        values = range(n_values)
        max_length = 0
        
        for val in values:
            encoding = self.encoder.encode(val)
            max_length = max(max_length, len(encoding))
            
        # 验证最大长度满足信息论下界
        import math
        theoretical_min = math.ceil(math.log2(n_values))
        self.assertGreaterEqual(
            max_length,
            theoretical_min,
            f"Max encoding length should be at least {theoretical_min}"
        )
        
    def test_string_encoding(self):
        """测试字符串编码"""
        test_string = "SRC"
        encoding = self.encoder.encode(test_string)
        
        # 验证编码结果
        self.assertIsInstance(encoding, str)
        self.assertTrue(all(c in '01' for c in encoding))
        self.assertGreater(len(encoding), 0)
        
        # 验证不同字符串产生不同编码
        another_string = "CRS"
        another_encoding = self.encoder.encode(another_string)
        self.assertNotEqual(encoding, another_encoding)
        
    def test_encoding_determinism(self):
        """测试编码确定性"""
        # 相同输入应该产生相同编码
        value = 42
        encoding1 = self.encoder.encode(value)
        encoding2 = self.encoder.encode(value)
        
        self.assertEqual(
            encoding1, encoding2,
            "Encoding should be deterministic"
        )
        
    def test_invalid_decoding(self):
        """测试无效解码处理"""
        # 测试解码无效二进制串
        invalid_strings = ["", "2", "10a", "xyz"]
        
        for invalid in invalid_strings:
            if invalid and all(c in '01' for c in invalid):
                # 有效的二进制串应该能解码（可能得到0）
                result = self.encoder.decode(invalid)
                self.assertIsInstance(result, int)
            else:
                # 无效输入返回0作为默认值
                result = self.encoder.decode(invalid)
                self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()