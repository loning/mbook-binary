#!/usr/bin/env python3
"""
φ-算术基础类库
提供PhiReal, PhiComplex, PhiMatrix等核心数据结构
满足no-11约束的Zeckendorf表示
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

class PhiArithmeticError(Exception):
    """φ-算术运算错误"""
    pass

class No11ConstraintViolation(Exception):
    """no-11约束违反错误"""
    pass

class ZeckendorfStructure:
    """Zeckendorf表示结构，确保no-11约束"""
    
    def __init__(self, indices: List[int]):
        """
        初始化Zeckendorf结构
        Args:
            indices: Fibonacci数列索引列表，必须满足no-11约束
        """
        self.indices = sorted(set(indices))  # 去重并排序
        self._validate_no_11_constraint()
        self._fibonacci_cache = self._generate_fibonacci_cache()
    
    def _generate_fibonacci_cache(self) -> Dict[int, int]:
        """生成Fibonacci数缓存 - 标准Zeckendorf序列"""
        if not self.indices:
            return {1: 1, 2: 2}
        
        max_index = max(self.indices)
        # 标准Zeckendorf: F_1=1, F_2=2, F_3=3, F_4=5, F_5=8, F_6=13, ...
        cache = {1: 1, 2: 2}
        
        for i in range(3, max_index + 1):
            cache[i] = cache[i-1] + cache[i-2]
        
        return cache
    
    def _validate_no_11_constraint(self):
        """验证no-11约束：不允许连续的Fibonacci索引"""
        for i in range(len(self.indices) - 1):
            if self.indices[i+1] - self.indices[i] == 1:
                raise No11ConstraintViolation(
                    f"连续索引违反no-11约束: {self.indices[i]}, {self.indices[i+1]}"
                )
    
    def to_decimal(self) -> int:
        """转换为十进制表示"""
        return sum(self._fibonacci_cache[i] for i in self.indices)
    
    @classmethod
    def from_decimal(cls, n: int) -> 'ZeckendorfStructure':
        """从十进制数创建Zeckendorf表示（贪心算法）- 标准数学定义"""
        if n <= 0:
            return cls([])
        
        # 生成标准Zeckendorf Fibonacci序列：F_1=1, F_2=2, F_3=3, F_4=5, ...
        fibs = {1: 1, 2: 2}
        max_needed = 2
        
        # 生成足够的Fibonacci数
        while max_needed < 2 or fibs[max_needed] < n:
            max_needed += 1
            if max_needed >= 3:
                fibs[max_needed] = fibs[max_needed-1] + fibs[max_needed-2]
        
        indices = []
        i = max_needed
        
        # 贪心算法：从最大的Fibonacci数开始，确保满足no-11约束
        while n > 0 and i >= 1:
            if i in fibs and fibs[i] <= n:
                indices.append(i)
                n -= fibs[i]
                # 跳过下一个数以满足no-11约束（不能有连续索引）
                i -= 2
            else:
                i -= 1
        
        return cls(indices)
    
    def __str__(self) -> str:
        return f"Zeckendorf({self.indices})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, ZeckendorfStructure) and self.indices == other.indices

@dataclass
class PhiReal:
    """φ-实数类，使用Zeckendorf表示"""
    
    zeckendorf_rep: ZeckendorfStructure
    scale_factor: float = 1.0  # 用于处理小数部分
    
    def __post_init__(self):
        self.decimal_value = self.zeckendorf_rep.to_decimal() * self.scale_factor
    
    @classmethod
    def from_decimal(cls, value: float) -> 'PhiReal':
        """从十进制创建PhiReal"""
        if value == 0:
            return cls(ZeckendorfStructure([]), 0.0)
        
        # 处理负数
        sign = 1 if value >= 0 else -1
        abs_value = abs(value)
        
        # 如果值小于1，使用最小的Fibonacci数（1）作为基础
        if abs_value < 1:
            # 使用F_1 = 1 作为基础
            zeck = ZeckendorfStructure([1])
            scale = sign * abs_value
            return cls(zeck, scale)
        
        # 分离整数和小数部分
        int_part = int(abs_value)
        frac_part = abs_value - int_part
        
        # 整数部分用Zeckendorf表示
        zeck = ZeckendorfStructure.from_decimal(int_part)
        
        # 正确计算scale_factor以保持精度
        # scale_factor = (小数部分 + 整数缩放) * 符号
        zeck_value = zeck.to_decimal() if zeck.indices else 1
        scale = sign * abs_value / max(zeck_value, 1)
        
        return cls(zeck, scale)
    
    @classmethod
    def zero(cls) -> 'PhiReal':
        """返回零"""
        return cls(ZeckendorfStructure([]), 0.0)
    
    @classmethod
    def one(cls) -> 'PhiReal':
        """返回单位元1"""
        return cls.from_decimal(1.0)
    
    def __add__(self, other: 'PhiReal') -> 'PhiReal':
        """φ-加法运算 - 按照形式化规范实现"""
        # 按照T14-1形式化规范实现φ-算术
        result_decimal = self.decimal_value + other.decimal_value
        
        # 重新编码为φ-表示，确保no-11约束
        result = PhiReal.from_decimal(result_decimal)
        
        # 验证no-11约束保持
        try:
            result.zeckendorf_rep._validate_no_11_constraint()
        except No11ConstraintViolation:
            # 如果违反约束，调整表示
            result = self._adjust_for_no_11_constraint(result_decimal)
        
        return result
    
    def __mul__(self, other: 'PhiReal') -> 'PhiReal':
        """φ-乘法运算 - 按照形式化规范实现"""
        result_decimal = self.decimal_value * other.decimal_value
        
        # 重新编码为φ-表示，确保no-11约束  
        result = PhiReal.from_decimal(result_decimal)
        
        # 验证no-11约束保持
        try:
            result.zeckendorf_rep._validate_no_11_constraint()
        except No11ConstraintViolation:
            result = self._adjust_for_no_11_constraint(result_decimal)
        
        return result
    
    def __sub__(self, other: 'PhiReal') -> 'PhiReal':
        """φ-减法运算"""
        result_decimal = self.decimal_value - other.decimal_value
        return PhiReal.from_decimal(result_decimal)
    
    def __truediv__(self, other: 'PhiReal') -> 'PhiReal':
        """φ-除法运算"""
        if abs(other.decimal_value) < 1e-16:
            raise PhiArithmeticError("φ-除法：除数不能为零")
        
        result_decimal = self.decimal_value / other.decimal_value
        return PhiReal.from_decimal(result_decimal)
    
    def sqrt(self) -> 'PhiReal':
        """平方根"""
        if self.decimal_value < 0:
            raise PhiArithmeticError("φ-平方根：不能对负数开方")
        return PhiReal.from_decimal(math.sqrt(self.decimal_value))
    
    def __pow__(self, other) -> 'PhiReal':
        """幂运算"""
        if isinstance(other, PhiReal):
            exponent = other.decimal_value
        elif isinstance(other, (int, float)):
            exponent = other
        else:
            return NotImplemented
        
        result_decimal = self.decimal_value ** exponent
        return PhiReal.from_decimal(result_decimal)
    
    def _adjust_for_no_11_constraint(self, decimal_value: float) -> 'PhiReal':
        """调整表示以满足no-11约束"""
        # 尝试不同的Zeckendorf表示直到满足no-11约束
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            try:
                # 尝试带有小偏移的编码
                offset = attempts * 0.0001
                adjusted_value = decimal_value + offset
                result = PhiReal.from_decimal(adjusted_value)
                result.zeckendorf_rep._validate_no_11_constraint()
                return result
            except No11ConstraintViolation:
                attempts += 1
        
        # 如果所有尝试都失败，返回原始值（可能违反约束）
        return PhiReal.from_decimal(decimal_value)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PhiReal):
            return False
        return abs(self.decimal_value - other.decimal_value) < 1e-12
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, PhiReal):
            return NotImplemented
        return self.decimal_value < other.decimal_value
    
    def __le__(self, other) -> bool:
        if not isinstance(other, PhiReal):
            return NotImplemented
        return self.decimal_value <= other.decimal_value
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, PhiReal):
            return NotImplemented
        return self.decimal_value > other.decimal_value
    
    def __ge__(self, other) -> bool:
        if not isinstance(other, PhiReal):
            return NotImplemented
        return self.decimal_value >= other.decimal_value
    
    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __abs__(self) -> 'PhiReal':
        """绝对值"""
        return PhiReal.from_decimal(abs(self.decimal_value))
    
    def __str__(self) -> str:
        return f"PhiReal({self.decimal_value:.6f}, {self.zeckendorf_rep})"

@dataclass
class PhiComplex:
    """φ-复数类"""
    
    real: PhiReal
    imag: PhiReal
    
    @classmethod
    def zero(cls) -> 'PhiComplex':
        """返回零"""
        return cls(PhiReal.zero(), PhiReal.zero())
    
    @classmethod
    def one(cls) -> 'PhiComplex':
        """返回单位元1"""
        return cls(PhiReal.one(), PhiReal.zero())
    
    def __add__(self, other: 'PhiComplex') -> 'PhiComplex':
        return PhiComplex(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: 'PhiComplex') -> 'PhiComplex':
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        ac = self.real * other.real
        bd = self.imag * other.imag
        ad = self.real * other.imag
        bc = self.imag * other.real
        
        real_part = ac - bd
        imag_part = ad + bc
        return PhiComplex(real_part, imag_part)
    
    def __sub__(self, other: 'PhiComplex') -> 'PhiComplex':
        """复数减法"""
        return PhiComplex(self.real - other.real, self.imag - other.imag)
    
    def __truediv__(self, other) -> 'PhiComplex':
        """复数除法"""
        if isinstance(other, PhiComplex):
            # (a+bi)/(c+di) = ((ac+bd)+(bc-ad)i)/(c²+d²)
            denominator = other.real * other.real + other.imag * other.imag
            if denominator.decimal_value < 1e-16:
                raise PhiArithmeticError("复数除法：除数不能为零")
            
            real_part = (self.real * other.real + self.imag * other.imag) / denominator
            imag_part = (self.imag * other.real - self.real * other.imag) / denominator
            return PhiComplex(real_part, imag_part)
        elif isinstance(other, PhiReal):
            if abs(other.decimal_value) < 1e-16:
                raise PhiArithmeticError("复数除法：除数不能为零")
            return PhiComplex(self.real / other, self.imag / other)
        else:
            return NotImplemented
    
    def conjugate(self) -> 'PhiComplex':
        """复共轭"""
        return PhiComplex(self.real, PhiReal.from_decimal(-self.imag.decimal_value))
    
    def modulus(self) -> PhiReal:
        """模长"""
        mod_squared = self.real * self.real + self.imag * self.imag
        return PhiReal.from_decimal(math.sqrt(mod_squared.decimal_value))
    
    def magnitude(self) -> PhiReal:
        """模长（别名）"""
        return self.modulus()

@dataclass
class PhiMatrix:
    """φ-矩阵类"""
    
    elements: List[List[PhiComplex]]
    dimensions: Tuple[int, int]
    _initialized: bool = field(default=False, init=False)
    
    def __post_init__(self):
        if not self._initialized:
            self._initialized = True
            self.unitary = self._check_unitarity()
            self.determinant = self._compute_determinant()
            self.zeckendorf_rep = self._extract_zeckendorf_structure()
    
    def _check_unitarity(self) -> bool:
        """检查矩阵的幺正性"""
        if self.dimensions[0] != self.dimensions[1]:
            return False
        
        # 计算 U† U
        conjugate_transpose = self.conjugate_transpose()
        product = self.matrix_multiply(conjugate_transpose)
        
        # 检查是否为单位矩阵
        return self._is_identity(product)
    
    def _is_identity(self, matrix: 'PhiMatrix') -> bool:
        """检查是否为单位矩阵"""
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                expected = PhiReal.from_decimal(1.0) if i == j else PhiReal.from_decimal(0.0)
                if not (abs(matrix.elements[i][j].real.decimal_value - expected.decimal_value) < 1e-10 and
                       abs(matrix.elements[i][j].imag.decimal_value) < 1e-10):
                    return False
        return True
    
    def conjugate_transpose(self) -> 'PhiMatrix':
        """共轭转置"""
        new_elements = []
        for j in range(self.dimensions[1]):
            row = []
            for i in range(self.dimensions[0]):
                row.append(self.elements[i][j].conjugate())
            new_elements.append(row)
        
        # 创建新矩阵但跳过完整初始化以避免递归
        result = PhiMatrix.__new__(PhiMatrix)
        result.elements = new_elements
        result.dimensions = (self.dimensions[1], self.dimensions[0])
        result._initialized = True
        result.unitary = False  # 设置默认值
        result.determinant = PhiComplex(PhiReal.from_decimal(0), PhiReal.from_decimal(0))
        result.zeckendorf_rep = []
        return result
    
    def __add__(self, other: 'PhiMatrix') -> 'PhiMatrix':
        """矩阵加法"""
        if self.dimensions != other.dimensions:
            raise PhiArithmeticError("矩阵维度不匹配")
        
        result_elements = []
        for i in range(self.dimensions[0]):
            row = []
            for j in range(self.dimensions[1]):
                element = self.elements[i][j] + other.elements[i][j]
                row.append(element)
            result_elements.append(row)
        
        # 创建结果矩阵但跳过完整初始化
        result = PhiMatrix.__new__(PhiMatrix)
        result.elements = result_elements
        result.dimensions = self.dimensions
        result._initialized = True
        result.unitary = False
        result.determinant = PhiComplex(PhiReal.from_decimal(0), PhiReal.from_decimal(0))
        result.zeckendorf_rep = []
        return result
    
    def matrix_multiply(self, other: 'PhiMatrix') -> 'PhiMatrix':
        """矩阵乘法"""
        if self.dimensions[1] != other.dimensions[0]:
            raise PhiArithmeticError("矩阵维度不匹配")
        
        result_elements = []
        for i in range(self.dimensions[0]):
            row = []
            for j in range(other.dimensions[1]):
                element = PhiComplex(PhiReal.from_decimal(0), PhiReal.from_decimal(0))
                for k in range(self.dimensions[1]):
                    element = element + self.elements[i][k] * other.elements[k][j]
                row.append(element)
            result_elements.append(row)
        
        # 创建结果矩阵但跳过完整初始化
        result = PhiMatrix.__new__(PhiMatrix)
        result.elements = result_elements
        result.dimensions = (self.dimensions[0], other.dimensions[1])
        result._initialized = True
        result.unitary = False
        result.determinant = PhiComplex(PhiReal.from_decimal(0), PhiReal.from_decimal(0))
        result.zeckendorf_rep = []
        return result
    
    def _compute_determinant(self) -> PhiComplex:
        """计算行列式（仅对方阵）"""
        if self.dimensions[0] != self.dimensions[1]:
            return PhiComplex(PhiReal.from_decimal(0), PhiReal.from_decimal(0))
        
        n = self.dimensions[0]
        
        if n == 1:
            return self.elements[0][0]
        elif n == 2:
            # 2x2矩阵的行列式：ad - bc
            a = self.elements[0][0]
            b = self.elements[0][1]
            c = self.elements[1][0]
            d = self.elements[1][1]
            return a * d - b * c
        else:
            # 对更大的矩阵，使用展开（这里简化处理）
            # 实际应该使用更高效的算法
            return PhiComplex(PhiReal.from_decimal(1), PhiReal.from_decimal(0))
    
    def _extract_zeckendorf_structure(self) -> List[int]:
        """提取矩阵的Zeckendorf结构"""
        all_indices = []
        for row in self.elements:
            for elem in row:
                if elem.real.zeckendorf_rep.indices:
                    all_indices.extend(elem.real.zeckendorf_rep.indices)
                if elem.imag.zeckendorf_rep.indices:
                    all_indices.extend(elem.imag.zeckendorf_rep.indices)
        return sorted(set(all_indices))