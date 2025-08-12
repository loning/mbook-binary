#!/usr/bin/env python3
"""
T{n}理论测试基类
基于zeckendorf库的统一测试框架
"""

import unittest
import sys
import math
from pathlib import Path
from typing import List, Optional

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入现有工具
from tools.theory_parser import TheoryParser, TheoryNode

# 尝试导入zeckendorf库
try:
    import zeckendorf
    HAS_ZECKENDORF = True
    print("✅ zeckendorf库已加载")
except ImportError:
    HAS_ZECKENDORF = False
    print("⚠️ zeckendorf库未安装，使用内置实现")


class BaseTheoryTest(unittest.TestCase):
    """
    T{n}理论测试基类
    
    提供基于zeckendorf库的统一测试框架，包含：
    - Zeckendorf分解验证
    - Fibonacci数列验证  
    - 黄金比例计算
    - 理论文档解析
    - 通用测试工具方法
    """
    
    # 阻止unittest直接运行基类
    __test__ = False
    
    @classmethod
    def setUpClass(cls):
        """初始化测试工具和常量"""
        cls.parser = TheoryParser()
        cls.theories_dir = Path(__file__).parent.parent / "theories"
        
        # 黄金比例
        cls.PHI = (1 + math.sqrt(5)) / 2
        
        # Fibonacci数列 (F1=1, F2=2, F3=3, F4=5, F5=8, ...)
        cls.FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        
        # Fibonacci集合，用于快速查找
        cls.FIB_SET = set(cls.FIBONACCI)
        
        print(f"🔧 测试环境初始化完成 (φ = {cls.PHI:.6f})")
    
    def get_theory_number(self) -> int:
        """
        返回理论编号 - 子类必须重写
        如果基类被直接调用，会跳过测试
        """
        if self.__class__ == BaseTheoryTest:
            self.skipTest("基类不应直接运行测试")
        raise NotImplementedError("子类必须实现 get_theory_number() 方法")
    
    # =================================
    # Zeckendorf相关方法
    # =================================
    
    def zeckendorf_decompose(self, n: int) -> List[int]:
        """
        计算数字n的Zeckendorf分解
        使用zeckendorf库（如果可用）或内置实现
        """
        if HAS_ZECKENDORF:
            try:
                # 使用zeckendorf库
                decomp = zeckendorf.zeckendorf_decompose(n)
                return sorted(decomp)
            except:
                # 如果库方法失败，回退到内置实现
                return self._builtin_zeckendorf_decompose(n)
        else:
            return self._builtin_zeckendorf_decompose(n)
    
    def _builtin_zeckendorf_decompose(self, n: int) -> List[int]:
        """内置Zeckendorf分解实现"""
        if n <= 0:
            return []
        
        result = []
        for fib in reversed(self.FIBONACCI):
            if fib <= n:
                result.append(fib)
                n -= fib
                if n == 0:
                    break
        
        return sorted(result)
    
    def verify_zeckendorf_properties(self, decomp: List[int]) -> bool:
        """
        验证Zeckendorf分解的性质：
        1. 所有数字都是Fibonacci数
        2. 没有连续的Fibonacci数
        3. 分解唯一性
        """
        if not decomp:
            return True
            
        # 检查所有数字都是Fibonacci数
        for num in decomp:
            if num not in self.FIB_SET:
                return False
        
        # 检查没有连续的Fibonacci数
        for i in range(len(decomp) - 1):
            curr_idx = self.FIBONACCI.index(decomp[i])
            next_idx = self.FIBONACCI.index(decomp[i + 1])
            if next_idx == curr_idx + 1:  # 连续的Fibonacci数
                return False
        
        return True
    
    # =================================
    # Fibonacci相关方法  
    # =================================
    
    def is_fibonacci_number(self, n: int) -> bool:
        """检查n是否为Fibonacci数"""
        return n in self.FIB_SET
    
    def fibonacci_index(self, fib_num: int) -> Optional[int]:
        """返回Fibonacci数的索引位置"""
        try:
            return self.FIBONACCI.index(fib_num) + 1  # F1, F2, F3, ...
        except ValueError:
            return None
    
    def golden_ratio_test(self, fn: int, fn1: int) -> float:
        """测试相邻Fibonacci数的比值是否接近黄金比例"""
        if fn == 0:
            return float('inf')
        return fn1 / fn
    
    # =================================
    # 理论文档相关方法
    # =================================
    
    def get_theory_file(self) -> Path:
        """获取理论文件路径"""
        theory_files = list(self.theories_dir.glob(f"T{self.get_theory_number()}__*.md"))
        self.assertTrue(len(theory_files) > 0, 
                       f"未找到T{self.get_theory_number()}的理论文件")
        return theory_files[0]
    
    def load_theory_node(self) -> TheoryNode:
        """加载并解析理论节点"""
        theory_file = self.get_theory_file()
        node = self.parser.parse_filename(theory_file.name)
        self.assertIsNotNone(node, f"无法解析理论文件: {theory_file.name}")
        return node
    
    # =================================
    # 信息论相关方法
    # =================================
    
    def phi_bits(self, n: int) -> float:
        """计算以φ为底的对数 (φ-bits)"""
        if n <= 0:
            return 0.0
        return math.log(n) / math.log(self.PHI)
    
    def shannon_bits(self, n: int) -> float:
        """计算以2为底的对数 (Shannon bits)"""
        if n <= 0:
            return 0.0
        return math.log2(n)
    
    def information_efficiency(self, n: int) -> float:
        """计算φ-bits相对于Shannon bits的效率"""
        shannon = self.shannon_bits(n)
        phi = self.phi_bits(n)
        if shannon == 0:
            return 1.0
        return phi / shannon
    
    # =================================
    # 通用验证方法
    # =================================
    
    def assert_zeckendorf_valid(self, decomp: List[int], msg: str = ""):
        """断言Zeckendorf分解有效"""
        self.assertTrue(self.verify_zeckendorf_properties(decomp), 
                       f"无效的Zeckendorf分解: {decomp}. {msg}")
    
    def assert_fibonacci_recursion(self, fn: int, fn_minus_1: int, fn_minus_2: int, msg: str = ""):
        """断言Fibonacci递归关系: Fn = F(n-1) + F(n-2)"""
        self.assertEqual(fn, fn_minus_1 + fn_minus_2,
                        f"Fibonacci递归失败: {fn} ≠ {fn_minus_1} + {fn_minus_2}. {msg}")
    
    def assert_golden_ratio_convergence(self, fn: int, fn_minus_1: int, tolerance: float = 0.1, msg: str = ""):
        """断言黄金比例收敛性"""
        if fn_minus_1 == 0:
            return  # 跳过除零情况
            
        ratio = fn / fn_minus_1
        diff = abs(ratio - self.PHI)
        self.assertLess(diff, tolerance,
                       f"黄金比例收敛失败: {ratio:.6f} 与 φ={self.PHI:.6f} 差异 {diff:.6f} > {tolerance}. {msg}")
    
    def assert_theory_consistency(self, theory_node: TheoryNode, msg: str = ""):
        """断言理论一致性"""
        # 基本属性检查
        self.assertEqual(theory_node.theory_number, self.get_theory_number(),
                        f"理论编号不匹配. {msg}")
        
        # Zeckendorf分解验证
        self.assert_zeckendorf_valid(theory_node.zeckendorf_decomp, 
                                    f"理论的Zeckendorf分解无效. {msg}")
        
        # 分解和应该等于理论编号
        decomp_sum = sum(theory_node.zeckendorf_decomp)
        self.assertEqual(decomp_sum, self.get_theory_number(),
                        f"Zeckendorf分解和 {decomp_sum} ≠ 理论编号 {self.get_theory_number()}. {msg}")
    
    # =================================
    # 数学验证方法
    # =================================
    
    def assert_approximately_equal(self, actual: float, expected: float, 
                                 tolerance: float = 1e-6, msg: str = ""):
        """断言浮点数近似相等"""
        diff = abs(actual - expected)
        self.assertLess(diff, tolerance,
                       f"数值不相等: {actual} ≠ {expected} (差异 {diff} > {tolerance}). {msg}")
    
    def assert_phi_relationship(self, value: float, phi_expression: str, 
                               tolerance: float = 0.01, msg: str = ""):
        """断言数值与黄金比例表达式的关系"""
        # 这里可以扩展来解析φ的表达式
        # 目前支持简单情况
        if phi_expression == "φ":
            expected = self.PHI
        elif phi_expression == "φ²":
            expected = self.PHI ** 2
        elif phi_expression == "1/φ":
            expected = 1 / self.PHI
        elif phi_expression == "1/φ²":
            expected = 1 / (self.PHI ** 2)
        else:
            # 尝试eval（仅用于测试）
            try:
                expected = eval(phi_expression.replace('φ', str(self.PHI)))
            except:
                self.fail(f"无法解析φ表达式: {phi_expression}")
        
        self.assert_approximately_equal(value, expected, tolerance,
                                       f"与φ关系验证失败: {value} ≠ {phi_expression} = {expected}. {msg}")


if __name__ == '__main__':
    # 基类不应直接运行
    print("📚 T{n}理论测试基类")
    print(f"✨ 支持zeckendorf库: {'是' if HAS_ZECKENDORF else '否'}")
    print(f"🔢 黄金比例 φ = {(1 + math.sqrt(5)) / 2:.10f}")
    print(f"📊 内置Fibonacci序列: {[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]}")
    print("⚠️  这是基类，请创建具体的理论测试类")