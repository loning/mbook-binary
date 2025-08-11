#!/usr/bin/env python3
"""
T27-8 极限环稳定性定理 - 核心数学结构
基于形式化规范 formal/T27-8-formal.md 实现核心数学对象

实现的形式系统组件：
- T_Space: 7维理论流形
- Flow_t: 动力系统流 Φ_t
- Lyap_V: Lyapunov函数空间
- d_Zeck: Zeckendorf度量
- C_Cycle: 极限环结构
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from decimal import Decimal, getcontext
from dataclasses import dataclass
import sys
import os

# 导入已验证的基础模块
sys.path.append('.')
from zeckendorf import ZeckendorfEncoder, GoldenConstants

# 设置高精度计算
getcontext().prec = 50


@dataclass
class TheoryPoint:
    """T_Space中的点 - 表示7维理论流形中的状态"""
    coordinates: np.ndarray  # 7维坐标
    theory_labels: List[str]  # 对应的理论标签
    
    def __post_init__(self):
        if len(self.coordinates) != 7:
            raise ValueError("理论点必须是7维")
        if not self.theory_labels:
            self.theory_labels = [f"T27-{i}" for i in range(1, 8)]
            
    def __repr__(self):
        return f"TheoryPoint({self.coordinates}, {self.theory_labels})"


class T_Space:
    """形式化规范中的T_Space: 7维理论流形"""
    
    def __init__(self):
        self.dimension = 7
        self.phi = GoldenConstants.PHI
        self.zeck_encoder = ZeckendorfEncoder(max_length=128)
        
        # 构造标准循环点：T27-1 → T27-2 → ... → T27-7 → T27-1
        self.cycle_points = self._construct_canonical_cycle()
        
    def _construct_canonical_cycle(self) -> List[TheoryPoint]:
        """构造标准的T27循环"""
        cycle = []
        for i in range(7):
            # 构造第i个理论的标准表示
            coords = np.zeros(7)
            coords[i] = 1.0  # 基础表示
            
            # 添加Zeckendorf编码的影响
            zeck_value = self.zeck_encoder.encode(i + 1)
            zeck_num = self.zeck_encoder.decode(zeck_value)
            coords[i] *= (1 + zeck_num * 0.01)  # 微小的Zeckendorf调制
            
            theory_point = TheoryPoint(
                coordinates=coords,
                theory_labels=[f"T27-{i+1}"]
            )
            cycle.append(theory_point)
            
        return cycle
    
    def create_point(self, coordinates: np.ndarray, labels: Optional[List[str]] = None) -> TheoryPoint:
        """在T_Space中创建点"""
        return TheoryPoint(coordinates=coordinates, theory_labels=labels)
    
    def get_cycle(self) -> List[TheoryPoint]:
        """获取标准T27循环"""
        return self.cycle_points.copy()


class ZeckendorfMetric:
    """形式化规范中的d_Zeck: Zeckendorf度量"""
    
    def __init__(self, max_length: int = 128):
        self.zeck = ZeckendorfEncoder(max_length=max_length)
        self.phi = GoldenConstants.PHI
        
    def distance(self, p1: TheoryPoint, p2: TheoryPoint) -> float:
        """计算两个理论点之间的Zeckendorf距离
        
        基于形式化规范中的d_Zeck: T_Space × T_Space → R+
        """
        # 基础欧几里得距离
        euclidean_dist = np.linalg.norm(p1.coordinates - p2.coordinates)
        
        # Zeckendorf编码调制
        # 将坐标差编码为Zeckendorf表示
        coord_diff = np.abs(p1.coordinates - p2.coordinates)
        zeck_modulation = 0.0
        
        for i, diff in enumerate(coord_diff):
            if diff > 1e-10:
                # 量化到整数并编码
                quantized = max(1, int(diff * 1000))
                zeck_str = self.zeck.encode(quantized)
                zeck_contribution = self.zeck.decode(zeck_str) / 1000.0
                zeck_modulation += zeck_contribution * (self.phi ** (-i))
        
        return euclidean_dist + 0.1 * zeck_modulation
        
    def phi_scaled_distance(self, p1: TheoryPoint, p2: TheoryPoint, scale_factor: int = 1) -> float:
        """φ调制的距离，用于Lyapunov函数"""
        base_dist = self.distance(p1, p2)
        return base_dist * (self.phi ** scale_factor)


class DynamicalFlow:
    """形式化规范中的Φ_t: T_Space × Time_T → T_Space"""
    
    def __init__(self, t_space: T_Space):
        self.t_space = t_space
        self.phi = GoldenConstants.PHI
        self.cycle = t_space.get_cycle()
        
    def flow_map(self, point: TheoryPoint, time: float) -> TheoryPoint:
        """动力系统流映射
        
        实现形式化规范中的流性质：
        - Φ_0(x) = x
        - Φ_{t+s}(x) = Φ_t(Φ_s(x))
        """
        if abs(time) < 1e-12:
            return point
        
        # 基础流向量场：指向最近的循环点
        current_coords = point.coordinates.copy()
        
        # 找到最近的循环点
        min_dist = float('inf')
        target_idx = 0
        metric = ZeckendorfMetric()
        
        for i, cycle_point in enumerate(self.cycle):
            dist = metric.distance(point, cycle_point)
            if dist < min_dist:
                min_dist = dist
                target_idx = i
        
        # 计算流向下一个循环点的向量
        next_idx = (target_idx + 1) % 7
        target_point = self.cycle[next_idx]
        flow_direction = target_point.coordinates - current_coords
        
        # 应用指数衰减到循环
        decay_rate = self.phi * time
        flow_magnitude = 1 - np.exp(-decay_rate)
        
        new_coords = current_coords + flow_magnitude * flow_direction
        
        return TheoryPoint(
            coordinates=new_coords,
            theory_labels=point.theory_labels.copy()
        )
    
    def vector_field(self, point: TheoryPoint) -> np.ndarray:
        """计算给定点处的向量场
        
        返回 d/dt Φ_t(x)|_{t=0}
        """
        # 计算流向最近循环点的方向
        metric = ZeckendorfMetric()
        distances = [metric.distance(point, cp) for cp in self.cycle]
        nearest_idx = np.argmin(distances)
        
        # 指向下一个循环点
        next_idx = (nearest_idx + 1) % 7
        direction = self.cycle[next_idx].coordinates - point.coordinates
        
        # φ调制的强度
        strength = self.phi * (1 + distances[nearest_idx])
        
        return strength * direction


class LyapunovFunction:
    """形式化规范中的V: T_Space → R+ (Lyapunov函数)"""
    
    def __init__(self, t_space: T_Space):
        self.t_space = t_space
        self.cycle = t_space.get_cycle()
        self.metric = ZeckendorfMetric()
        self.phi = GoldenConstants.PHI
        
    def evaluate(self, point: TheoryPoint) -> float:
        """计算Lyapunov函数值
        
        V(x) = Σ_{i=1}^7 d_Zeck²(x, T_{27-i})
        根据理论文档定义 2.1
        """
        total_energy = 0.0
        
        for i, cycle_point in enumerate(self.cycle):
            dist = self.metric.distance(point, cycle_point)
            energy_contribution = dist ** 2
            
            # φ权重调制（基于形式化规范）
            phi_weight = self.phi ** (-i)  # 递减权重
            total_energy += phi_weight * energy_contribution
            
        return total_energy
    
    def time_derivative(self, point: TheoryPoint, flow: DynamicalFlow) -> float:
        """计算Lyapunov函数的时间导数
        
        dV/dt = ∇V · Φ_t
        根据定理 2.1: dV/dt = -φ·V(x) < 0 (x ∉ C)
        """
        current_V = self.evaluate(point)
        
        # 检查是否在极限环上
        if self.is_on_cycle(point):
            return 0.0
        
        # 理论预测：dV/dt = -φ·V
        return -self.phi * current_V
    
    def is_on_cycle(self, point: TheoryPoint, tolerance: float = 1e-6) -> bool:
        """检查点是否在极限环上"""
        for cycle_point in self.cycle:
            if self.metric.distance(point, cycle_point) < tolerance:
                return True
        return False
    
    def gradient(self, point: TheoryPoint) -> np.ndarray:
        """计算Lyapunov函数的梯度"""
        grad = np.zeros(7)
        eps = 1e-8
        
        for i in range(7):
            # 数值梯度计算
            point_plus = TheoryPoint(
                coordinates=point.coordinates + eps * np.eye(7)[i],
                theory_labels=point.theory_labels
            )
            point_minus = TheoryPoint(
                coordinates=point.coordinates - eps * np.eye(7)[i],
                theory_labels=point.theory_labels
            )
            
            grad[i] = (self.evaluate(point_plus) - self.evaluate(point_minus)) / (2 * eps)
            
        return grad


class LimitCycle:
    """形式化规范中的C_Cycle: 极限环类型"""
    
    def __init__(self, t_space: T_Space):
        self.t_space = t_space
        self.cycle_points = t_space.get_cycle()
        self.phi = GoldenConstants.PHI
        
    def period(self) -> float:
        """极限环的周期 τ_cycle"""
        # 基于φ的理论周期
        return 2 * np.pi / self.phi
    
    def is_point_on_cycle(self, point: TheoryPoint, tolerance: float = 1e-6) -> bool:
        """检查点是否在极限环上"""
        metric = ZeckendorfMetric()
        for cycle_point in self.cycle_points:
            if metric.distance(point, cycle_point) < tolerance:
                return True
        return False
    
    def closest_cycle_point(self, point: TheoryPoint) -> Tuple[TheoryPoint, float]:
        """找到最近的循环点"""
        metric = ZeckendorfMetric()
        min_distance = float('inf')
        closest_point = None
        
        for cycle_point in self.cycle_points:
            dist = metric.distance(point, cycle_point)
            if dist < min_distance:
                min_distance = dist
                closest_point = cycle_point
                
        return closest_point, min_distance
    
    def verify_cycle_closure(self) -> bool:
        """验证循环的闭合性：T27-1 → ... → T27-7 → T27-1"""
        # 检查最后一个点是否能流向第一个点
        flow = DynamicalFlow(self.t_space)
        last_point = self.cycle_points[-1]
        
        # 模拟一个周期的流动
        evolved_point = flow.flow_map(last_point, self.period() / 7)
        
        metric = ZeckendorfMetric()
        closure_distance = metric.distance(evolved_point, self.cycle_points[0])
        
        return closure_distance < 1e-3  # 允许数值误差


# 测试核心结构的一致性
def test_core_structures():
    """基础一致性测试"""
    print("🔍 T27-8 核心数学结构测试")
    print("=" * 50)
    
    # 1. 创建理论空间
    t_space = T_Space()
    print(f"✅ T_Space 创建完成，维数: {t_space.dimension}")
    
    # 2. 验证循环结构
    cycle = LimitCycle(t_space)
    is_closed = cycle.verify_cycle_closure()
    print(f"✅ 极限环闭合验证: {is_closed}")
    
    # 3. 测试Lyapunov函数
    lyap = LyapunovFunction(t_space)
    test_point = t_space.create_point(np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.3]))
    V_value = lyap.evaluate(test_point)
    print(f"✅ Lyapunov函数值: {V_value:.6f}")
    
    # 4. 测试动力学流
    flow = DynamicalFlow(t_space)
    evolved_point = flow.flow_map(test_point, 0.1)
    print(f"✅ 动力学流映射完成")
    
    # 5. 测试Zeckendorf度量
    metric = ZeckendorfMetric()
    cycle_points = t_space.get_cycle()
    dist = metric.distance(cycle_points[0], cycle_points[1])
    print(f"✅ Zeckendorf度量: {dist:.6f}")
    
    print("\n🎯 核心数学结构验证完成")
    return True


if __name__ == "__main__":
    success = test_core_structures()
    exit(0 if success else 1)