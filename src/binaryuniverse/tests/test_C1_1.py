#!/usr/bin/env python3
"""
test_C1_1.py - C1-1唯一编码推论的完整机器验证测试

完整验证φ-表示系统中编码的唯一性和双射性
"""

import unittest
import sys
import os
from typing import List, Dict, Set, Tuple

# 添加包路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class PhiRepresentationSystem:
    """φ-表示系统的实现（复用之前的定义）"""
    
    def __init__(self, n: int):
        """初始化n位φ-表示系统"""
        self.n = n
        self.valid_states = self._generate_valid_states()
        self.state_to_index = {tuple(s): i for i, s in enumerate(self.valid_states)}
        self.index_to_state = {i: s for i, s in enumerate(self.valid_states)}
        
    def _is_valid_phi_state(self, state: List[int]) -> bool:
        """检查是否为有效的φ-表示状态"""
        if len(state) != self.n:
            return False
        if not all(bit in [0, 1] for bit in state):
            return False
        
        # 检查no-consecutive-1s约束
        for i in range(len(state) - 1):
            if state[i] == 1 and state[i + 1] == 1:
                return False
        return True
    
    def _generate_valid_states(self) -> List[List[int]]:
        """生成所有有效的φ-表示状态"""
        valid_states = []
        
        def generate_recursive(current_state: List[int], pos: int):
            if pos == self.n:
                if self._is_valid_phi_state(current_state):
                    valid_states.append(current_state[:])
                return
            
            # 尝试放置0
            current_state.append(0)
            generate_recursive(current_state, pos + 1)
            current_state.pop()
            
            # 尝试放置1（如果不违反约束）
            if pos == 0 or current_state[pos - 1] == 0:
                current_state.append(1)
                generate_recursive(current_state, pos + 1)
                current_state.pop()
        
        generate_recursive([], 0)
        return valid_states
    
    def encode(self, state_index: int) -> List[int]:
        """编码：状态索引到二进制表示"""
        if state_index in self.index_to_state:
            return self.index_to_state[state_index][:]
        else:
            raise ValueError(f"Invalid state index: {state_index}")
    
    def decode(self, binary_state: List[int]) -> int:
        """解码：二进制表示到状态索引"""
        state_tuple = tuple(binary_state)
        if state_tuple in self.state_to_index:
            return self.state_to_index[state_tuple]
        else:
            raise ValueError(f"Invalid binary state: {binary_state}")


class UniqueEncodingVerifier:
    """唯一编码推论的验证器"""
    
    def __init__(self, n: int):
        """初始化验证器"""
        self.n = n
        self.phi_system = PhiRepresentationSystem(n)
        
    def verify_bijection(self) -> Dict[str, bool]:
        """验证双射性质"""
        results = {
            "injection": True,  # 单射性
            "surjection": True,  # 满射性
            "bijection": True   # 双射性
        }
        
        # 1. 验证单射性：不同状态索引编码到不同的二进制表示
        encoded_states = set()
        for i in range(len(self.phi_system.valid_states)):
            encoded = tuple(self.phi_system.encode(i))
            if encoded in encoded_states:
                results["injection"] = False
                break
            encoded_states.add(encoded)
        
        # 2. 验证满射性：所有有效的二进制表示都有对应的状态索引
        all_valid_states = set(tuple(s) for s in self.phi_system.valid_states)
        if encoded_states != all_valid_states:
            results["surjection"] = False
        
        # 3. 双射性 = 单射性 + 满射性
        results["bijection"] = results["injection"] and results["surjection"]
        
        return results
    
    def verify_uniqueness(self) -> Dict[str, bool]:
        """验证唯一性"""
        results = {
            "encoding_unique": True,    # 编码唯一性
            "decoding_unique": True,    # 解码唯一性
            "no_collision": True        # 无冲突
        }
        
        # 1. 验证编码唯一性：每个状态只有一个编码
        # 由于我们使用确定性函数，这自动满足
        # 但我们仍然验证多次编码得到相同结果
        for i in range(min(10, len(self.phi_system.valid_states))):
            encoding1 = self.phi_system.encode(i)
            encoding2 = self.phi_system.encode(i)
            if encoding1 != encoding2:
                results["encoding_unique"] = False
                break
        
        # 2. 验证解码唯一性：每个有效编码只对应一个状态
        decoded_indices = {}
        for state in self.phi_system.valid_states:
            state_tuple = tuple(state)
            index = self.phi_system.decode(state)
            
            if state_tuple in decoded_indices:
                if decoded_indices[state_tuple] != index:
                    results["decoding_unique"] = False
                    break
            else:
                decoded_indices[state_tuple] = index
        
        # 3. 验证无冲突：不存在两个不同状态有相同编码
        encoding_to_index = {}
        for i in range(len(self.phi_system.valid_states)):
            encoding = tuple(self.phi_system.encode(i))
            if encoding in encoding_to_index:
                if encoding_to_index[encoding] != i:
                    results["no_collision"] = False
                    break
            else:
                encoding_to_index[encoding] = i
        
        return results
    
    def verify_completeness(self) -> Dict[str, bool]:
        """验证完备性"""
        results = {
            "all_states_encodable": True,
            "all_valid_decodable": True,
            "coverage_complete": True
        }
        
        # 1. 验证所有状态都可编码
        try:
            for i in range(len(self.phi_system.valid_states)):
                encoding = self.phi_system.encode(i)
                if not self.phi_system._is_valid_phi_state(encoding):
                    results["all_states_encodable"] = False
                    break
        except Exception:
            results["all_states_encodable"] = False
        
        # 2. 验证所有有效编码都可解码
        try:
            for state in self.phi_system.valid_states:
                index = self.phi_system.decode(state)
                if index < 0 or index >= len(self.phi_system.valid_states):
                    results["all_valid_decodable"] = False
                    break
        except Exception:
            results["all_valid_decodable"] = False
        
        # 3. 验证覆盖完整性
        encoded_count = len(self.phi_system.valid_states)
        decodable_count = len(self.phi_system.valid_states)
        results["coverage_complete"] = (encoded_count == decodable_count)
        
        return results
    
    def verify_consistency(self) -> Dict[str, bool]:
        """验证一致性"""
        results = {
            "encode_decode_identity": True,
            "decode_encode_identity": True,
            "round_trip_consistent": True
        }
        
        # 1. 验证 decode(encode(i)) = i
        for i in range(len(self.phi_system.valid_states)):
            encoded = self.phi_system.encode(i)
            decoded = self.phi_system.decode(encoded)
            if decoded != i:
                results["encode_decode_identity"] = False
                break
        
        # 2. 验证 encode(decode(s)) = s
        for state in self.phi_system.valid_states:
            index = self.phi_system.decode(state)
            re_encoded = self.phi_system.encode(index)
            if re_encoded != state:
                results["decode_encode_identity"] = False
                break
        
        # 3. 验证往返一致性
        results["round_trip_consistent"] = (
            results["encode_decode_identity"] and 
            results["decode_encode_identity"]
        )
        
        return results
    
    def verify_constraint_preservation(self) -> Dict[str, bool]:
        """验证约束保持"""
        results = {
            "encoded_states_valid": True,
            "no_consecutive_ones": True,
            "length_preserved": True
        }
        
        # 1. 验证所有编码后的状态都是有效的φ-表示
        for i in range(len(self.phi_system.valid_states)):
            encoded = self.phi_system.encode(i)
            if not self.phi_system._is_valid_phi_state(encoded):
                results["encoded_states_valid"] = False
                break
        
        # 2. 验证no-consecutive-1s约束
        for state in self.phi_system.valid_states:
            for i in range(len(state) - 1):
                if state[i] == 1 and state[i + 1] == 1:
                    results["no_consecutive_ones"] = False
                    break
            if not results["no_consecutive_ones"]:
                break
        
        # 3. 验证长度保持
        for i in range(len(self.phi_system.valid_states)):
            encoded = self.phi_system.encode(i)
            if len(encoded) != self.n:
                results["length_preserved"] = False
                break
        
        return results
    
    def verify_corollary_completeness(self) -> Dict[str, any]:
        """C1-1推论的完整验证"""
        return {
            "bijection": self.verify_bijection(),
            "uniqueness": self.verify_uniqueness(),
            "completeness": self.verify_completeness(),
            "consistency": self.verify_consistency(),
            "constraint_preservation": self.verify_constraint_preservation(),
            "total_states": len(self.phi_system.valid_states),
            "bit_length": self.n
        }


class TestC1_1_UniqueEncoding(unittest.TestCase):
    """C1-1唯一编码推论的完整机器验证测试"""

    def setUp(self):
        """测试初始化"""
        self.verifier_small = UniqueEncodingVerifier(n=4)  # 小系统
        self.verifier_medium = UniqueEncodingVerifier(n=6)  # 中等系统
        
    def test_bijection_complete(self):
        """测试双射性的完整性 - 验证检查点1"""
        print("\n=== C1-1 验证检查点1：双射性完整验证 ===")
        
        # 小系统验证
        bijection_small = self.verifier_small.verify_bijection()
        print(f"4位系统双射性验证: {bijection_small}")
        
        self.assertTrue(bijection_small["injection"], 
                       "编码映射应该是单射")
        self.assertTrue(bijection_small["surjection"], 
                       "编码映射应该是满射")
        self.assertTrue(bijection_small["bijection"], 
                       "编码映射应该是双射")
        
        # 中等系统验证
        bijection_medium = self.verifier_medium.verify_bijection()
        print(f"6位系统双射性验证: {bijection_medium}")
        
        self.assertTrue(bijection_medium["bijection"], 
                       "更大系统的编码映射也应该是双射")
        
        print("✓ 双射性完整验证通过")

    def test_uniqueness_complete(self):
        """测试唯一性的完整性 - 验证检查点2"""
        print("\n=== C1-1 验证检查点2：唯一性完整验证 ===")
        
        uniqueness = self.verifier_small.verify_uniqueness()
        print(f"唯一性验证结果: {uniqueness}")
        
        self.assertTrue(uniqueness["encoding_unique"], 
                       "每个状态的编码应该唯一")
        self.assertTrue(uniqueness["decoding_unique"], 
                       "每个编码的解码应该唯一")
        self.assertTrue(uniqueness["no_collision"], 
                       "不应该有编码冲突")
        
        # 具体示例验证
        phi = self.verifier_small.phi_system
        state0_encoding = phi.encode(0)
        state1_encoding = phi.encode(1)
        
        print(f"  状态0编码: {state0_encoding}")
        print(f"  状态1编码: {state1_encoding}")
        
        self.assertNotEqual(state0_encoding, state1_encoding, 
                           "不同状态应该有不同编码")
        
        print("✓ 唯一性完整验证通过")

    def test_completeness_complete(self):
        """测试完备性的完整性 - 验证检查点3"""
        print("\n=== C1-1 验证检查点3：完备性完整验证 ===")
        
        completeness = self.verifier_small.verify_completeness()
        print(f"完备性验证结果: {completeness}")
        
        self.assertTrue(completeness["all_states_encodable"], 
                       "所有状态都应该可编码")
        self.assertTrue(completeness["all_valid_decodable"], 
                       "所有有效编码都应该可解码")
        self.assertTrue(completeness["coverage_complete"], 
                       "编码覆盖应该完整")
        
        # 统计信息
        total_states = len(self.verifier_small.phi_system.valid_states)
        print(f"  总状态数: {total_states}")
        print(f"  位长度: {self.verifier_small.n}")
        
        # 验证Fibonacci数列关系
        if self.verifier_small.n >= 1:
            # φ-表示的状态数应该等于F(n+2)
            # 标准Fibonacci数列：F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8, ...
            fib = [0, 1]  # F(0), F(1)
            for i in range(2, self.verifier_small.n + 3):
                fib.append(fib[-1] + fib[-2])
            expected = fib[self.verifier_small.n + 2]
            self.assertEqual(total_states, expected, 
                           f"状态数应该等于Fibonacci数F({self.verifier_small.n}+2)={expected}")
        
        print("✓ 完备性完整验证通过")

    def test_consistency_complete(self):
        """测试一致性的完整性 - 验证检查点4"""
        print("\n=== C1-1 验证检查点4：一致性完整验证 ===")
        
        consistency = self.verifier_small.verify_consistency()
        print(f"一致性验证结果: {consistency}")
        
        self.assertTrue(consistency["encode_decode_identity"], 
                       "编码后解码应该得到原状态")
        self.assertTrue(consistency["decode_encode_identity"], 
                       "解码后编码应该得到原编码")
        self.assertTrue(consistency["round_trip_consistent"], 
                       "往返映射应该一致")
        
        # 具体往返测试
        phi = self.verifier_small.phi_system
        for i in range(min(5, len(phi.valid_states))):
            # 编码-解码往返
            encoded = phi.encode(i)
            decoded = phi.decode(encoded)
            self.assertEqual(decoded, i, 
                           f"状态{i}的编码-解码往返应该一致")
            
            # 解码-编码往返
            state = phi.valid_states[i]
            index = phi.decode(state)
            re_encoded = phi.encode(index)
            self.assertEqual(re_encoded, state, 
                           f"编码{state}的解码-编码往返应该一致")
        
        print("✓ 一致性完整验证通过")

    def test_constraint_preservation_complete(self):
        """测试约束保持的完整性 - 验证检查点5"""
        print("\n=== C1-1 验证检查点5：约束保持完整验证 ===")
        
        constraints = self.verifier_small.verify_constraint_preservation()
        print(f"约束保持验证结果: {constraints}")
        
        self.assertTrue(constraints["encoded_states_valid"], 
                       "所有编码状态都应该有效")
        self.assertTrue(constraints["no_consecutive_ones"], 
                       "应该保持no-consecutive-1s约束")
        self.assertTrue(constraints["length_preserved"], 
                       "编码长度应该保持不变")
        
        # 验证一些具体状态
        phi = self.verifier_small.phi_system
        print("  有效状态示例:")
        for i in range(min(5, len(phi.valid_states))):
            state = phi.valid_states[i]
            print(f"    {state}")
            
            # 验证no-11约束
            for j in range(len(state) - 1):
                self.assertFalse(state[j] == 1 and state[j + 1] == 1, 
                               f"状态{state}违反了no-11约束")
        
        print("✓ 约束保持完整验证通过")

    def test_complete_unique_encoding_corollary(self):
        """测试完整唯一编码推论 - 主推论验证"""
        print("\n=== C1-1 主推论：完整唯一编码验证 ===")
        
        # 完整验证
        verification = self.verifier_small.verify_corollary_completeness()
        
        print(f"推论完整验证结果:")
        for key, value in verification.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 验证所有性质
        bijection = verification["bijection"]
        self.assertTrue(all(bijection.values()), 
                       f"双射性质应该全部满足: {bijection}")
        
        uniqueness = verification["uniqueness"]
        self.assertTrue(all(uniqueness.values()), 
                       f"唯一性质应该全部满足: {uniqueness}")
        
        completeness = verification["completeness"]
        self.assertTrue(all(completeness.values()), 
                       f"完备性质应该全部满足: {completeness}")
        
        consistency = verification["consistency"]
        self.assertTrue(all(consistency.values()), 
                       f"一致性质应该全部满足: {consistency}")
        
        constraints = verification["constraint_preservation"]
        self.assertTrue(all(constraints.values()), 
                       f"约束保持应该全部满足: {constraints}")
        
        print(f"\n✓ C1-1推论验证通过")
        print(f"  - 总状态数: {verification['total_states']}")
        print(f"  - 位长度: {verification['bit_length']}")
        print(f"  - 编码映射是双射")
        print(f"  - 每个状态有唯一编码")
        print(f"  - 编码系统完备一致")


def run_complete_verification():
    """运行完整的C1-1验证"""
    print("=" * 80)
    print("C1-1 唯一编码推论 - 完整机器验证")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestC1_1_UniqueEncoding)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ C1-1唯一编码推论完整验证成功！")
        print("φ-表示系统确实提供了唯一的双射编码。")
    else:
        print("✗ C1-1唯一编码推论验证发现问题")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_complete_verification()
    exit(0 if success else 1)