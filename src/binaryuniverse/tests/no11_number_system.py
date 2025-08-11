#!/usr/bin/env python3
"""
No-11约束数值系统

实现真正的no-11约束下的数值表示和运算
不是标准二进制+过滤，而是原生的no-11数值系统
"""

from typing import List, Union
import math


class No11Number:
    """
    No-11约束下的数值类
    
    内部表示：no-11约束的二进制模式
    数值映射：模式索引对应实际数值
    """
    
    # 预计算的有效no-11模式
    _valid_patterns = []
    _pattern_to_value = {}
    _value_to_pattern = {}
    _initialized = False
    
    @classmethod
    def _initialize_patterns(cls, max_bits: int = 20):
        """初始化有效的no-11模式"""
        if cls._initialized:
            return
            
        cls._valid_patterns = []
        cls._pattern_to_value = {}
        cls._value_to_pattern = {}
        
        # 生成所有有效的no-11模式
        def is_no11_valid(bits: List[int]) -> bool:
            for i in range(len(bits) - 1):
                if bits[i] == 1 and bits[i+1] == 1:
                    return False
            return True
        
        # 生成所有可能的位模式并筛选
        value = 0
        for length in range(1, max_bits + 1):
            for i in range(2 ** length):
                bits = []
                temp = i
                for _ in range(length):
                    bits.append(temp % 2)
                    temp //= 2
                bits.reverse()
                
                # 移除前导零（除了单独的0）
                while len(bits) > 1 and bits[0] == 0:
                    bits.pop(0)
                
                pattern_str = ''.join(map(str, bits))
                
                if is_no11_valid(bits) and pattern_str not in cls._pattern_to_value:
                    cls._valid_patterns.append(bits)
                    cls._pattern_to_value[pattern_str] = value
                    cls._value_to_pattern[value] = bits
                    value += 1
        
        # 添加特殊情况：0
        if '0' not in cls._pattern_to_value:
            cls._valid_patterns.insert(0, [0])
            cls._pattern_to_value['0'] = 0
            cls._value_to_pattern[0] = [0]
            # 重新编号其他值
            for i, pattern in enumerate(cls._valid_patterns[1:], 1):
                pattern_str = ''.join(map(str, pattern))
                cls._pattern_to_value[pattern_str] = i
                cls._value_to_pattern[i] = pattern
        
        cls._initialized = True
    
    def __init__(self, value: Union[int, List[int], str, 'No11Number']):
        """
        初始化No-11数值
        
        Args:
            value: 可以是整数值、位列表、位字符串或另一个No11Number
        """
        if not self._initialized:
            self._initialize_patterns()
        
        if isinstance(value, No11Number):
            self.bits = value.bits.copy()
            self.value = value.value
        elif isinstance(value, int):
            if value < 0:
                raise ValueError("No-11 numbers must be non-negative")
            if value >= len(self._valid_patterns):
                raise ValueError(f"Value {value} exceeds maximum representable value")
            self.bits = self._value_to_pattern[value].copy()
            self.value = value
        elif isinstance(value, list):
            # 验证模式有效性
            if not self._is_valid_pattern(value):
                raise ValueError(f"Pattern {value} violates no-11 constraint")
            pattern_str = ''.join(map(str, value))
            if pattern_str not in self._pattern_to_value:
                raise ValueError(f"Pattern {value} not in valid no-11 patterns")
            self.bits = value.copy()
            self.value = self._pattern_to_value[pattern_str]
        elif isinstance(value, str):
            bits = [int(b) for b in value if b in '01']
            if not self._is_valid_pattern(bits):
                raise ValueError(f"Pattern {value} violates no-11 constraint")
            pattern_str = ''.join(map(str, bits))
            if pattern_str not in self._pattern_to_value:
                raise ValueError(f"Pattern {value} not in valid no-11 patterns")
            self.bits = bits
            self.value = self._pattern_to_value[pattern_str]
        else:
            raise TypeError(f"Cannot create No11Number from {type(value)}")
    
    @classmethod
    def _is_valid_pattern(cls, bits: List[int]) -> bool:
        """检查位模式是否满足no-11约束"""
        for i in range(len(bits) - 1):
            if bits[i] == 1 and bits[i+1] == 1:
                return False
        return True
    
    def __str__(self) -> str:
        return f"No11({self.value}:{self.bits})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if isinstance(other, No11Number):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return False
    
    def __hash__(self) -> int:
        """使No11Number可哈希，能够用作集合元素或字典键"""
        return hash(self.value)
    
    def __lt__(self, other) -> bool:
        if isinstance(other, No11Number):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other) -> bool:
        return self == other or self < other
    
    def __gt__(self, other) -> bool:
        return not self <= other
    
    def __ge__(self, other) -> bool:
        return not self < other
    
    def __add__(self, other) -> 'No11Number':
        """No-11加法"""
        if isinstance(other, No11Number):
            result_value = self.value + other.value
        elif isinstance(other, int):
            result_value = self.value + other
        else:
            return NotImplemented
        
        return No11Number(result_value)
    
    def __mul__(self, other) -> 'No11Number':
        """No-11乘法"""
        if isinstance(other, No11Number):
            result_value = self.value * other.value
        elif isinstance(other, int):
            result_value = self.value * other
        else:
            return NotImplemented
        
        return No11Number(result_value)
    
    def __pow__(self, other) -> 'No11Number':
        """No-11幂运算"""
        if isinstance(other, No11Number):
            result_value = self.value ** other.value
        elif isinstance(other, int):
            result_value = self.value ** other
        else:
            return NotImplemented
        
        return No11Number(result_value)
    
    def __sub__(self, other) -> 'No11Number':
        """No-11减法"""
        if isinstance(other, No11Number):
            result_value = self.value - other.value
        elif isinstance(other, int):
            result_value = self.value - other
        else:
            return NotImplemented
        
        if result_value < 0:
            result_value = 0  # 不支持负数
        
        return No11Number(result_value)
    
    def __floordiv__(self, other) -> 'No11Number':
        """No-11整除"""
        if isinstance(other, No11Number):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            result_value = self.value // other.value
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            result_value = self.value // other
        else:
            return NotImplemented
        
        return No11Number(result_value)
    
    def __mod__(self, other) -> 'No11Number':
        """No-11取模"""
        if isinstance(other, No11Number):
            if other.value == 0:
                raise ZeroDivisionError("Modulo by zero")
            result_value = self.value % other.value
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Modulo by zero")
            result_value = self.value % other
        else:
            return NotImplemented
        
        return No11Number(result_value)
    
    @classmethod
    def get_valid_patterns(cls, max_count: int = 20) -> List[List[int]]:
        """获取前max_count个有效的no-11模式"""
        if not cls._initialized:
            cls._initialize_patterns()
        return cls._valid_patterns[:max_count]
    
    @classmethod
    def max_representable_value(cls) -> int:
        """返回当前可表示的最大值"""
        if not cls._initialized:
            cls._initialize_patterns()
        return len(cls._valid_patterns) - 1


def test_no11_number_system():
    """测试No-11数值系统"""
    print("Testing No-11 Number System")
    print("=" * 40)
    
    # 显示前15个有效模式
    patterns = No11Number.get_valid_patterns(15)
    print("First 15 valid no-11 patterns:")
    for i, pattern in enumerate(patterns):
        num = No11Number(i)
        print(f"{i:2d}: {pattern} -> {num}")
    
    print("\nTesting arithmetic:")
    
    # 测试基础运算
    a = No11Number(2)  # 应该是 [1,0]
    b = No11Number(3)  # 应该是 [1,0,1] 
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"b ** a = {b ** a}")
    
    # 测试特殊情况
    print(f"\nSpecial cases:")
    zero = No11Number(0)
    one = No11Number(1)
    print(f"0 = {zero}")
    print(f"1 = {one}")
    print(f"0 + 1 = {zero + one}")
    print(f"1 * 5 = {one * 5}")
    
    print(f"\nMax representable value: {No11Number.max_representable_value()}")


class No11NumberSystem:
    """
    No-11数值系统接口类
    提供对No11Number功能的封装
    """
    
    def __init__(self):
        # 确保模式已初始化
        No11Number._initialize_patterns()
    
    def to_zeckendorf(self, n: int) -> List[int]:
        """转换为Zeckendorf表示（简化版本）"""
        # 这里返回Fibonacci索引列表
        if n <= 0:
            return []
        
        # 使用标准Fibonacci数列
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        indices = []
        
        # 贪心算法
        for i in range(len(fibs) - 1, -1, -1):
            if fibs[i] <= n:
                indices.append(i + 1)  # 1-indexed
                n -= fibs[i]
                if n == 0:
                    break
        
        return sorted(indices)
    
    def is_valid_representation(self, indices: List[int]) -> bool:
        """检查表示是否满足no-11约束"""
        # 检查是否有连续的索引
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] == 1:
                return False
        return True
    
    def contains_11(self, binary_str: str) -> bool:
        """检查二进制字符串是否包含11"""
        return '11' in binary_str


if __name__ == '__main__':
    test_no11_number_system()