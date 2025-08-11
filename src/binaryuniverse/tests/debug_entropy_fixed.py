#!/usr/bin/env python3
"""调试修正后的熵增计算"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phi_arithmetic import PhiReal
from no11_number_system import No11NumberSystem
from test_T17_3 import *

def debug_fixed_entropy_calculation():
    """调试修正后的熵增计算过程"""
    no11 = No11NumberSystem()
    algorithm = PhiMTheoryUnificationAlgorithm(no11)
    entropy_verifier = EntropyIncreaseVerifier(no11)
    
    # 创建测试理论
    string_theories = list(algorithm.duality_network.string_theories.values())
    unified_m_theory = PhiMTheorySpacetime()
    
    print("=== 修正后的熵增计算调试 ===")
    
    # 计算初始熵
    print("\n1. 计算各个弦理论的熵:")
    initial_entropy = PhiReal.zero()
    for i, theory in enumerate(string_theories):
        theory_entropy = entropy_verifier._compute_theory_entropy(theory)
        print(f"  {theory.name}: {theory_entropy.decimal_value:.6f}")
        initial_entropy += theory_entropy
    
    print(f"初始总熵: {initial_entropy.decimal_value:.6f}")
    
    # 计算M理论完整熵
    print("\n2. 计算M理论完整熵（包含所有组成部分）:")
    
    # 分别计算各个组成部分
    preservation_entropy = PhiReal.zero()
    for theory in string_theories:
        preservation_entropy += entropy_verifier._compute_theory_entropy(theory)
    
    relation_entropy = entropy_verifier._compute_duality_network_entropy(string_theories)
    mapping_entropy = entropy_verifier._compute_unification_mapping_entropy(string_theories)
    no11_encoding_entropy = entropy_verifier._compute_no11_encoding_entropy(unified_m_theory)
    self_reference_entropy = entropy_verifier._compute_self_reference_entropy(string_theories)
    
    print(f"  保存熵（必须包含所有原始信息）: {preservation_entropy.decimal_value:.6f}")
    print(f"  关系网络熵（对偶关系描述）: {relation_entropy.decimal_value:.6f}")
    print(f"  映射算法熵（11D→10D）: {mapping_entropy.decimal_value:.6f}")
    print(f"  no-11编码熵（约束满足）: {no11_encoding_entropy.decimal_value:.6f}")
    print(f"  自指描述熵（元理论）: {self_reference_entropy.decimal_value:.6f}")
    
    total_m_theory_entropy = (preservation_entropy + relation_entropy + 
                             mapping_entropy + no11_encoding_entropy + 
                             self_reference_entropy)
    
    print(f"M理论总熵: {total_m_theory_entropy.decimal_value:.6f}")
    
    # 计算熵增
    print("\n3. 熵增验证:")
    entropy_increase = total_m_theory_entropy - initial_entropy
    print(f"熵增量: {entropy_increase.decimal_value:.6f}")
    print(f"熵增？: {entropy_increase.decimal_value > 0}")
    
    # 解释结果
    print("\n4. 理论意义:")
    if entropy_increase.decimal_value > 0:
        print("✓ 熵增验证通过！")
        print("✓ 统一过程确实是复杂化，而非简化")
        print("✓ M理论必须包含:")
        print("  - 所有原始弦理论信息（保存熵）")
        print("  - 所有对偶关系的描述（关系熵）")
        print("  - 统一映射算法（映射熵）")
        print("  - no-11约束编码（编码熵）")
        print("  - 自指描述结构（自指熵）")
        print("✓ 这正是唯一公理'自指完备系统必然熵增'的体现")
    else:
        print("✗ 熵增验证失败")
        
    print(f"\n关键比例:")
    print(f"  保存熵/初始熵 = {preservation_entropy.decimal_value/initial_entropy.decimal_value:.2f}")
    print(f"  额外熵/初始熵 = {(total_m_theory_entropy.decimal_value - preservation_entropy.decimal_value)/initial_entropy.decimal_value:.2f}")

if __name__ == "__main__":
    debug_fixed_entropy_calculation()