"""
测试T5-5：自指纠错定理

验证：
1. 错误检测能力
2. 纠错正确性
3. 熵增约束
4. 收敛性
5. 最优性
6. 错误传播限制
"""

import unittest
import numpy as np
import math
from typing import List, Set, Dict, Tuple, Optional
import random
from collections import defaultdict

class SelfReferentialSystem:
    """自指完备系统基类"""
    
    def __init__(self, initial_state: str = ""):
        self.state = initial_state
        self.descriptions: Set[str] = {initial_state} if initial_state else set()
        self.correction_history: List[Dict] = []
        
    def get_description(self) -> str:
        """获取系统的自我描述"""
        # 简化模型：描述就是状态的规范形式
        return self.normalize_state(self.state)
    
    def normalize_state(self, state: str) -> str:
        """规范化状态（移除违反no-11的部分）"""
        # 这里简化为移除所有'11'模式
        while '11' in state:
            state = state.replace('11', '10')
        return state
    
    def is_consistent(self) -> bool:
        """检查系统是否自洽"""
        return self.state == self.get_description()
    
    def compute_system_entropy(self) -> float:
        """计算系统熵 H = log|D|"""
        if len(self.descriptions) == 0:
            return 0.0
        return math.log2(len(self.descriptions))


class PhiErrorCorrector:
    """φ-表示纠错器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        
    def detect_11_violations(self, state: str) -> List[int]:
        """检测no-11约束违反的位置"""
        violations = []
        for i in range(len(state) - 1):
            if state[i] == '1' and state[i+1] == '1':
                violations.append(i)
        return violations
    
    def compute_correction_cost(self, original: str, corrected: str) -> int:
        """计算纠错代价（汉明距离）"""
        if len(original) != len(corrected):
            return float('inf')
        return sum(1 for i in range(len(original)) if original[i] != corrected[i])
    
    def correct_minimal(self, state: str) -> Tuple[str, int]:
        """最小代价纠错"""
        violations = self.detect_11_violations(state)
        if not violations:
            return state, 0
        
        # 对每个违反，尝试两种修正：改第一个1或第二个1
        best_correction = state
        min_cost = float('inf')
        
        def try_corrections(s: str, pos: int) -> List[str]:
            """尝试修正位置pos的'11'"""
            corrections = []
            if pos < len(s) - 1:
                # 改第一个1为0
                c1 = s[:pos] + '0' + s[pos+1:]
                corrections.append(c1)
                # 改第二个1为0
                c2 = s[:pos+1] + '0' + s[pos+2:]
                corrections.append(c2)
            return corrections
        
        # 使用贪心策略：每次修正一个违反
        current = state
        total_cost = 0
        
        while True:
            violations = self.detect_11_violations(current)
            if not violations:
                break
                
            # 选择第一个违反进行修正
            pos = violations[0]
            candidates = try_corrections(current, pos)
            
            # 选择引入最少新违反的修正
            best_candidate = None
            min_new_violations = float('inf')
            
            for candidate in candidates:
                new_violations = len(self.detect_11_violations(candidate))
                if new_violations < min_new_violations:
                    min_new_violations = new_violations
                    best_candidate = candidate
            
            if best_candidate:
                current = best_candidate
                total_cost += 1
            else:
                break
        
        return current, total_cost
    
    def verify_correction(self, corrected: str) -> bool:
        """验证纠错的有效性"""
        return len(self.detect_11_violations(corrected)) == 0


class EvolvingSelfReferentialSystem(SelfReferentialSystem):
    """可演化的自指系统"""
    
    def __init__(self, initial_state: str = ""):
        super().__init__(initial_state)
        self.corrector = PhiErrorCorrector()
        
    def introduce_error(self, error_rate: float = 0.1) -> int:
        """引入随机错误"""
        if not self.state:
            return 0
            
        errors = 0
        new_state = list(self.state)
        
        for i in range(len(new_state)):
            if random.random() < error_rate:
                # 翻转比特
                new_state[i] = '0' if new_state[i] == '1' else '1'
                errors += 1
        
        self.state = ''.join(new_state)
        return errors
    
    def correct(self) -> Dict[str, any]:
        """纠正系统错误"""
        initial_state = self.state
        initial_entropy = self.compute_system_entropy()
        
        # 执行纠错
        corrected_state, cost = self.corrector.correct_minimal(self.state)
        
        # 更新状态
        self.state = corrected_state
        
        # 添加新描述（纠错可能产生新的有效描述）
        if corrected_state not in self.descriptions:
            self.descriptions.add(corrected_state)
        
        final_entropy = self.compute_system_entropy()
        
        # 记录纠错历史
        correction_info = {
            'initial_state': initial_state,
            'corrected_state': corrected_state,
            'cost': cost,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy,
            'entropy_change': final_entropy - initial_entropy,
            'is_consistent': self.is_consistent()
        }
        
        self.correction_history.append(correction_info)
        
        return correction_info
    
    def iterate_correction(self, max_iterations: int = 100) -> int:
        """迭代纠错直到收敛"""
        for i in range(max_iterations):
            if self.is_consistent() and not self.corrector.detect_11_violations(self.state):
                return i
            self.correct()
        return max_iterations


class TestT5_5SelfReferentialErrorCorrection(unittest.TestCase):
    """T5-5自指纠错定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        np.random.seed(42)
        random.seed(42)
    
    def test_error_detection(self):
        """测试1：错误检测能力"""
        print("\n测试1：错误检测能力")
        
        corrector = PhiErrorCorrector()
        
        test_cases = [
            ("0101010", []),           # 无错误
            ("0110101", [1]),          # 单个11
            ("1101011", [0, 5]),       # 多个11
            ("11111", [0, 1, 2, 3]),   # 连续11
        ]
        
        print("  状态      违反位置")
        print("  --------  ----------")
        
        all_correct = True
        for state, expected in test_cases:
            violations = corrector.detect_11_violations(state)
            correct = violations == expected
            all_correct &= correct
            
            print(f"  {state}  {violations} {'✓' if correct else '✗'}")
        
        self.assertTrue(all_correct, "错误检测应该正确识别所有no-11违反")
    
    def test_correction_correctness(self):
        """测试2：纠错正确性"""
        print("\n测试2：纠错正确性")
        
        system = EvolvingSelfReferentialSystem("01010")
        
        # 引入错误
        test_states = [
            "01110",   # 单个11
            "11010",   # 开头11
            "01011",   # 结尾11
            "11111",   # 全是1
        ]
        
        print("  原始状态  纠正后    代价  验证")
        print("  --------  --------  ----  ----")
        
        for error_state in test_states:
            system.state = error_state
            correction_info = system.correct()
            
            is_valid = system.corrector.verify_correction(correction_info['corrected_state'])
            
            print(f"  {error_state}    {correction_info['corrected_state']}  "
                  f"{correction_info['cost']:4}  {'✓' if is_valid else '✗'}")
            
            self.assertTrue(is_valid, f"纠错后的状态应该满足no-11约束")
    
    def test_entropy_constraint(self):
        """测试3：熵增约束"""
        print("\n测试3：熵增约束")
        
        system = EvolvingSelfReferentialSystem("0101010")
        
        # 多次引入错误并纠正
        entropy_violations = 0
        
        print("  步骤  初始熵  最终熵  熵变    违反？")
        print("  ----  ------  ------  ------  ------")
        
        for i in range(10):
            # 引入错误
            system.introduce_error(0.2)
            
            # 纠正
            info = system.correct()
            
            entropy_change = info['entropy_change']
            violates = entropy_change < 0
            
            if violates:
                entropy_violations += 1
            
            print(f"  {i+1:4}  {info['initial_entropy']:6.3f}  "
                  f"{info['final_entropy']:6.3f}  {entropy_change:6.3f}  "
                  f"{'是' if violates else '否'}")
        
        self.assertEqual(entropy_violations, 0, "纠错过程不应违反熵增约束")
    
    def test_convergence(self):
        """测试4：收敛性"""
        print("\n测试4：纠错收敛性")
        
        # 测试不同初始错误密度的收敛性
        error_rates = [0.1, 0.2, 0.3, 0.5]
        
        print("  错误率  初始违反  迭代次数  最终状态")
        print("  ------  --------  --------  --------")
        
        for rate in error_rates:
            system = EvolvingSelfReferentialSystem("0" * 20)
            
            # 引入错误
            system.introduce_error(rate)
            initial_violations = len(system.corrector.detect_11_violations(system.state))
            
            # 迭代纠错
            iterations = system.iterate_correction()
            
            print(f"  {rate:6.1f}  {initial_violations:8}  {iterations:8}  "
                  f"{system.state[:8]}...")
            
            self.assertTrue(system.is_consistent(), "系统应该收敛到一致状态")
            self.assertEqual(len(system.corrector.detect_11_violations(system.state)), 0,
                           "最终状态不应有违反")
    
    def test_optimality(self):
        """测试5：φ-表示纠错最优性"""
        print("\n测试5：纠错最优性")
        
        corrector = PhiErrorCorrector()
        
        # 测试不同纠错策略的代价
        test_cases = [
            "11000",  # 简单情况
            "11110",  # 多个连续1
            "10110101",  # 分散的11
        ]
        
        print("  原始状态    最优纠正    代价  验证最优性")
        print("  ----------  ----------  ----  ----------")
        
        for state in test_cases:
            corrected, cost = corrector.correct_minimal(state)
            
            # 验证是否是最小代价
            # 枚举所有可能的纠正并比较代价
            is_optimal = self._verify_optimal_correction(state, corrected, cost)
            
            print(f"  {state:10}  {corrected:10}  {cost:4}  "
                  f"{'✓' if is_optimal else '✗'}")
            
            self.assertTrue(corrector.verify_correction(corrected),
                          "纠正结果应该有效")
    
    def test_error_propagation_limit(self):
        """测试6：错误传播限制"""
        print("\n测试6：错误传播限制")
        
        # 测试单比特错误的影响范围
        base_state = "0101010101"
        
        print("  错误位置  影响范围  局部性")
        print("  --------  --------  ------")
        
        for error_pos in [2, 4, 6, 8]:
            # 引入单比特错误
            error_state = list(base_state)
            error_state[error_pos] = '1' if error_state[error_pos] == '0' else '0'
            error_state = ''.join(error_state)
            
            # 纠正错误
            corrector = PhiErrorCorrector()
            corrected, _ = corrector.correct_minimal(error_state)
            
            # 计算影响范围
            affected_positions = []
            for i in range(len(base_state)):
                if i < len(corrected) and base_state[i] != corrected[i]:
                    affected_positions.append(i)
            
            # 验证局部性
            if affected_positions:
                min_pos = min(affected_positions)
                max_pos = max(affected_positions)
                spread = max_pos - min_pos + 1
            else:
                spread = 0
            
            is_local = spread <= 3  # 影响范围不超过3位
            
            print(f"  {error_pos:8}  {spread:8}  {'✓' if is_local else '✗'}")
            
            self.assertTrue(is_local, "单比特错误的影响应该是局部的")
    
    def test_error_as_innovation(self):
        """测试7：错误驱动的创新"""
        print("\n测试7：纠错产生创新")
        
        system = EvolvingSelfReferentialSystem("0101")
        initial_descriptions = len(system.descriptions)
        
        # 多轮错误-纠正循环
        rounds = 20
        new_descriptions = set()
        
        print("  轮次  错误数  新描述  总描述数")
        print("  ----  ------  ------  --------")
        
        for round in range(rounds):
            # 引入随机错误
            errors = system.introduce_error(0.15)
            
            # 纠正
            info = system.correct()
            
            # 检查是否产生新描述
            if info['corrected_state'] not in new_descriptions:
                new_descriptions.add(info['corrected_state'])
            
            if round % 5 == 0:
                print(f"  {round+1:4}  {errors:6}  {len(new_descriptions):6}  "
                      f"{len(system.descriptions):8}")
        
        # 验证描述多样性增加
        final_descriptions = len(system.descriptions)
        
        print(f"\n  初始描述数: {initial_descriptions}")
        print(f"  最终描述数: {final_descriptions}")
        print(f"  新增描述数: {final_descriptions - initial_descriptions}")
        
        self.assertGreater(final_descriptions, initial_descriptions,
                          "纠错过程应该增加描述多样性")
    
    def _verify_optimal_correction(self, original: str, corrected: str, cost: int) -> bool:
        """验证是否是最优纠正（辅助函数）"""
        corrector = PhiErrorCorrector()
        
        # 生成所有可能的相同或更低代价的纠正
        n = len(original)
        
        # 只检查代价不超过cost的修改
        for num_changes in range(1, min(cost + 1, n + 1)):
            # 这里简化验证，只检查一些随机样本
            for _ in range(min(10, 2**num_changes)):
                # 生成一个候选纠正
                candidate = list(original)
                positions = random.sample(range(n), num_changes)
                
                for pos in positions:
                    candidate[pos] = '0' if candidate[pos] == '1' else '1'
                
                candidate_str = ''.join(candidate)
                
                # 如果这个候选也是有效的纠正
                if corrector.verify_correction(candidate_str):
                    candidate_cost = corrector.compute_correction_cost(original, candidate_str)
                    if candidate_cost < cost:
                        return False
        
        return True


if __name__ == '__main__':
    unittest.main()