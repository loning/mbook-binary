#!/usr/bin/env python3
"""
系统组件基础库
提供二进制宇宙理论中的系统组件模型
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from zeckendorf import ZeckendorfEncoder, GoldenConstants, EntropyCalculator


@dataclass
class SystemComponent:
    """系统组件基础类"""
    
    id: int
    length: int  # 二进制串长度
    current_entropy: float  # 当前熵
    time_scale: float  # 特征时间尺度
    capacity: float = 0.0  # 熵容量（自动计算）
    
    def __post_init__(self):
        """初始化后自动计算容量"""
        encoder = ZeckendorfEncoder()
        self.capacity = encoder.compute_capacity(self.length)
        
    def saturation(self) -> float:
        """计算饱和度"""
        return self.current_entropy / self.capacity if self.capacity > 0 else 0.0
        
    def max_rate(self) -> float:
        """计算最大熵增速率"""
        return self.capacity / self.time_scale
        
    def available_capacity(self) -> float:
        """计算可用容量"""
        return max(0, self.capacity - self.current_entropy)
        
    def is_saturated(self, threshold: float = None) -> bool:
        """判断是否饱和"""
        if threshold is None:
            threshold = GoldenConstants.PHI_INVERSE
        return self.saturation() > threshold


class SystemEvolver:
    """系统演化器基类"""
    
    def __init__(self, components: List[SystemComponent]):
        """
        初始化系统演化器
        
        Args:
            components: 系统组件列表
        """
        self.phi = GoldenConstants.PHI
        self.phi_inverse = GoldenConstants.PHI_INVERSE
        self.components = components
        self.encoder = ZeckendorfEncoder()
        self.time = 0.0
        self.entropy_history = []
        
    def identify_bottleneck(self) -> Tuple[int, float]:
        """
        识别瓶颈组件
        
        Returns:
            (瓶颈组件索引, 最大饱和度)
        """
        saturations = [comp.saturation() for comp in self.components]
        bottleneck_idx = np.argmax(saturations)
        max_saturation = saturations[bottleneck_idx]
        return bottleneck_idx, max_saturation
        
    def compute_system_entropy(self) -> float:
        """计算系统总熵"""
        return sum(comp.current_entropy for comp in self.components)
        
    def compute_max_entropy_rate(self) -> float:
        """
        计算系统最大熵增速率（木桶原理）
        
        Returns:
            受瓶颈限制的最大速率
        """
        rates = [comp.max_rate() for comp in self.components]
        return min(rates)
        
    def compute_effective_rate(self) -> float:
        """
        计算有效熵增速率（考虑饱和效应）
        
        Returns:
            考虑饱和衰减后的有效速率
        """
        bottleneck_idx, saturation = self.identify_bottleneck()
        max_rate = self.compute_max_entropy_rate()
        
        # 当饱和度超过φ^{-1}时，速率指数衰减
        if saturation > self.phi_inverse:
            reduction_factor = np.exp(-self.phi * saturation)
            return max_rate * reduction_factor
        return max_rate
        
    def apply_saturation_effect(self, rate: float, saturation: float) -> float:
        """
        应用饱和效应
        
        Args:
            rate: 原始速率
            saturation: 饱和度
            
        Returns:
            衰减后的速率
        """
        if saturation > self.phi_inverse:
            return rate * np.exp(-self.phi * saturation)
        return rate
        
    def evolve(self, dt: float) -> Dict:
        """
        演化系统一个时间步
        
        Args:
            dt: 时间步长
            
        Returns:
            演化状态字典
        """
        # 记录初始状态
        initial_entropy = self.compute_system_entropy()
        bottleneck_idx, saturation = self.identify_bottleneck()
        
        # 计算有效熵增速率
        effective_rate = self.compute_effective_rate()
        actual_system_rate = effective_rate
        
        # 系统总熵增
        total_delta_h = actual_system_rate * dt
        
        # 按组件容量比例分配熵增
        total_available = sum(comp.available_capacity() for comp in self.components)
        
        if total_available > 0:
            for comp in self.components:
                # 按可用容量比例分配
                weight = comp.available_capacity() / total_available
                delta_h = total_delta_h * weight
                
                # 确保不超过组件容量
                delta_h = min(delta_h, comp.available_capacity())
                
                # Fibonacci量子化（仅对显著变化）
                if delta_h > 0.01:
                    # 量子化到Fibonacci数的百分之一
                    delta_h_quantized = self.encoder.fibonacci_quantize(delta_h * 100) / 100.0
                    delta_h = min(delta_h_quantized, delta_h)
                    
                # 更新组件熵
                comp.current_entropy = min(comp.current_entropy + delta_h, comp.capacity)
        
        # 更新时间
        self.time += dt
        
        # 记录历史
        final_entropy = self.compute_system_entropy()
        state = {
            'time': self.time,
            'total_entropy': final_entropy,
            'entropy_rate': (final_entropy - initial_entropy) / dt if dt > 0 else 0,
            'bottleneck_idx': bottleneck_idx,
            'bottleneck_saturation': saturation,
            'effective_rate': effective_rate
        }
        self.entropy_history.append(state)
        
        return state
        
    def add_parallel_path(self, bottleneck_idx: int) -> None:
        """
        添加并行路径以突破瓶颈
        
        Args:
            bottleneck_idx: 瓶颈组件索引
        """
        bottleneck = self.components[bottleneck_idx]
        # 创建并行组件，容量加倍
        parallel = SystemComponent(
            id=len(self.components),
            length=bottleneck.length * 2,  # 双倍容量
            current_entropy=bottleneck.current_entropy * 0.5,  # 分担一半熵
            time_scale=bottleneck.time_scale
        )
        # 原组件熵减半（熵转移到并行路径）
        bottleneck.current_entropy *= 0.5
        self.components.append(parallel)
        
    def get_system_state(self) -> Dict:
        """获取系统当前状态"""
        return {
            'time': self.time,
            'total_entropy': self.compute_system_entropy(),
            'num_components': len(self.components),
            'bottleneck': self.identify_bottleneck(),
            'max_rate': self.compute_max_entropy_rate(),
            'effective_rate': self.compute_effective_rate(),
            'components': [
                {
                    'id': comp.id,
                    'capacity': comp.capacity,
                    'entropy': comp.current_entropy,
                    'saturation': comp.saturation()
                }
                for comp in self.components
            ]
        }


class BottleneckSystem(SystemEvolver):
    """木桶原理瓶颈系统"""
    
    def __init__(self, components: List[SystemComponent]):
        """专门用于研究瓶颈效应的系统"""
        super().__init__(components)
        
    def analyze_bottleneck(self) -> Dict:
        """分析瓶颈详情"""
        bottleneck_idx, saturation = self.identify_bottleneck()
        bottleneck = self.components[bottleneck_idx]
        
        # 计算瓶颈影响
        system_rate = self.compute_max_entropy_rate()
        without_bottleneck = float('inf')
        
        for i, comp in enumerate(self.components):
            if i != bottleneck_idx:
                without_bottleneck = min(without_bottleneck, comp.max_rate())
                
        bottleneck_impact = (without_bottleneck - system_rate) / without_bottleneck if without_bottleneck < float('inf') else 1.0
        
        return {
            'bottleneck_id': bottleneck.id,
            'bottleneck_idx': bottleneck_idx,
            'saturation': saturation,
            'capacity': bottleneck.capacity,
            'current_entropy': bottleneck.current_entropy,
            'max_rate': bottleneck.max_rate(),
            'is_critical': saturation > self.phi_inverse,
            'bottleneck_impact': bottleneck_impact,
            'system_rate_limited_to': system_rate
        }
        
    def predict_saturation_time(self) -> float:
        """预测达到临界饱和的时间"""
        bottleneck_idx, current_saturation = self.identify_bottleneck()
        
        if current_saturation >= self.phi_inverse:
            return 0.0  # 已经饱和
            
        bottleneck = self.components[bottleneck_idx]
        rate = self.compute_effective_rate()
        
        if rate <= 0:
            return float('inf')  # 永不饱和
            
        # 估算达到临界饱和所需的熵增
        critical_entropy = bottleneck.capacity * self.phi_inverse
        needed_entropy = critical_entropy - bottleneck.current_entropy
        
        # 考虑速率会随饱和度变化，使用积分估算
        estimated_time = needed_entropy / rate
        
        return estimated_time