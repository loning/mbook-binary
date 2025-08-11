#!/usr/bin/env python3
"""调试熵增计算"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phi_arithmetic import PhiReal
from no11_number_system import No11NumberSystem
from test_T17_3 import *

def debug_entropy_calculation():
    """调试熵增计算过程"""
    no11 = No11NumberSystem()
    algorithm = PhiMTheoryUnificationAlgorithm(no11)
    entropy_verifier = EntropyIncreaseVerifier(no11)
    
    # 创建测试理论
    string_theories = list(algorithm.duality_network.string_theories.values())
    unified_m_theory = PhiMTheorySpacetime()
    
    print("=== 熵增计算调试 ===")
    
    # 计算初始熵
    print("\n1. 计算各个弦理论的熵:")
    initial_entropy = PhiReal.zero()
    for i, theory in enumerate(string_theories):
        theory_entropy = entropy_verifier._compute_theory_entropy(theory)
        print(f"  {theory.name}: {theory_entropy.decimal_value:.6f}")
        initial_entropy += theory_entropy
    
    print(f"初始总熵: {initial_entropy.decimal_value:.6f}")
    
    # 计算M理论熵
    print("\n2. 计算M理论熵:")
    final_entropy = entropy_verifier._compute_m_theory_entropy(unified_m_theory)
    print(f"M理论熵: {final_entropy.decimal_value:.6f}")
    
    # 计算统一过程熵
    print("\n3. 计算统一过程熵:")
    unification_entropy = entropy_verifier._compute_unification_process_entropy(string_theories)
    print(f"统一过程熵: {unification_entropy.decimal_value:.6f}")
    
    # 计算总变化
    print("\n4. 熵变计算:")
    total_final = final_entropy + unification_entropy
    entropy_change = total_final - initial_entropy
    print(f"最终总熵: {total_final.decimal_value:.6f}")
    print(f"熵增量: {entropy_change.decimal_value:.6f}")
    print(f"熵增？: {entropy_change.decimal_value > 0}")
    
    # 分析问题
    print("\n5. 问题分析:")
    if entropy_change.decimal_value <= 0:
        print("问题：熵没有增加！")
        print("可能原因：")
        print(f"  - M理论熵({final_entropy.decimal_value:.6f}) + 统一熵({unification_entropy.decimal_value:.6f}) = {total_final.decimal_value:.6f}")
        print(f"  - 但初始熵: {initial_entropy.decimal_value:.6f}")
        print(f"  - 差值: {entropy_change.decimal_value:.6f}")
        
        if final_entropy.decimal_value < initial_entropy.decimal_value:
            print("  主要问题：M理论熵比弦理论总熵小")
        if unification_entropy.decimal_value < 0:
            print("  问题：统一过程熵为负")
    else:
        print("熵增验证通过！")

if __name__ == "__main__":
    debug_entropy_calculation()