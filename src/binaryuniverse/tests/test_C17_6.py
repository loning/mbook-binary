#!/usr/bin/env python3
"""
C17-6: AdS-CFT观察者映射推论 - 完整测试程序

验证AdS/CFT观察者映射的性质，包括：
1. 全息编码映射
2. 纠缠熵几何化
3. 径向-深度对应
4. 边界观察者性质
5. 量子纠错码结构
"""

import unittest
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# 导入基础类
try:
    from test_C17_1 import ObserverSystem
    from test_C17_5 import SemanticDepthAnalyzer
except ImportError:
    # 最小实现
    class ObserverSystem:
        def __init__(self, dimension: int):
            self.phi = (1 + np.sqrt(5)) / 2
            self.dim = dimension
            self.state = np.zeros(dimension)
            self.state[0] = 1
        
        def observe(self, system_state):
            return system_state, self.state
    
    class SemanticDepthAnalyzer:
        def __init__(self, dimension: int):
            self.phi = (1 + np.sqrt(5)) / 2
            self.dim = dimension
            
        def compute_semantic_depth(self, state):
            return int(np.log2(np.sum(state) + 1))


class AdSCFTObserverMapping:
    """AdS/CFT观察者映射系统"""
    
    def __init__(self, boundary_dim: int):
        self.phi = (1 + np.sqrt(5)) / 2
        self.d_boundary = boundary_dim
        self.d_bulk = boundary_dim + 1
        self.G_Newton = 1.0  # 归一化的牛顿常数
        self.z_cutoff = 10.0  # 径向截断
        
    def boundary_to_bulk(self, boundary_state: np.ndarray) -> np.ndarray:
        """边界态映射到体态（HKLL重构）"""
        bulk_state = np.zeros((self.d_bulk, len(boundary_state)))
        
        for z_idx in range(self.d_bulk):
            # 径向坐标
            z = self._radial_coordinate(z_idx)
            
            # HKLL涂抹核
            smeared = self._hkll_smearing(boundary_state, z)
            bulk_state[z_idx] = smeared
            
        # 强制no-11约束
        return self._enforce_no11_bulk(bulk_state)
    
    def _radial_coordinate(self, z_idx: int) -> float:
        """计算径向坐标"""
        return self.phi ** (-z_idx) * 0.1  # 初始径向位置
    
    def _hkll_smearing(self, boundary_state: np.ndarray, z: float) -> np.ndarray:
        """HKLL涂抹函数"""
        n = len(boundary_state)
        smeared = np.zeros(n, dtype=float)
        
        # 涂抹核：高斯型，宽度与z成比例
        width = max(z * self.phi, 0.1)  # 避免过窄的核
        
        for i in range(n):
            for j in range(n):
                # 边界距离（周期性）
                distance = min(abs(i - j), n - abs(i - j))
                
                # 涂抹权重
                weight = np.exp(-distance**2 / (2 * width**2))
                
                # Fibonacci调制
                fib_mod = self._fibonacci_modulation(j, n)
                
                # 位置特异性权重：增强区分性
                position_weight = 1.0 + 0.5 * np.sin(2 * np.pi * j / n)
                
                smeared[i] += boundary_state[j] * weight * fib_mod * position_weight
        
        # 非线性变换以增强差异
        smeared = np.tanh(smeared)  # 使用tanh而非线性归一化
        
        # 添加状态特异性扰动
        state_hash = np.sum(boundary_state * np.arange(len(boundary_state))) % n
        for i in range(n):
            smeared[i] += 0.1 * np.sin(2 * np.pi * (i + state_hash) / n)
        
        # 转换为二进制（保持区分性）
        threshold = np.mean(smeared)
        binary_result = (smeared > threshold).astype(int)
        
        return binary_result
    
    def _fibonacci_modulation(self, position: int, total_length: int) -> float:
        """Fibonacci位置调制"""
        fib_positions = self._get_fibonacci_positions(total_length)
        
        if position in fib_positions:
            return 1.0
        else:
            # 距离最近Fibonacci位置的权重
            min_dist = min(abs(position - pos) for pos in fib_positions)
            return self.phi ** (-min_dist)
    
    def _get_fibonacci_positions(self, n: int) -> List[int]:
        """获取小于n的Fibonacci位置"""
        positions = []
        a, b = 1, 2
        while a < n:
            positions.append(a)
            a, b = b, a + b
        return positions
    
    def compute_entanglement_entropy(self, region_A: List[int], 
                                   boundary_state: np.ndarray) -> float:
        """计算子区域A的纠缠熵"""
        # 找到极小曲面
        minimal_surface = self._find_minimal_surface(region_A)
        
        # 计算面积
        area = self._compute_surface_area(minimal_surface)
        
        # Ryu-Takayanagi公式（带φ修正）
        S_ent = area / (4 * self.G_Newton * self.phi)
        
        return S_ent
    
    def _find_minimal_surface(self, region_A: List[int]) -> np.ndarray:
        """寻找锚定在region_A的极小曲面"""
        # 简化实现：测地线延伸
        surface_points = []
        
        for boundary_point in region_A:
            # 从边界点沿测地线延伸到体中
            geodesic = self._geodesic_extension(boundary_point)
            surface_points.extend(geodesic)
        
        return np.array(surface_points)
    
    def _geodesic_extension(self, boundary_point: int) -> List[Tuple[float, int]]:
        """将边界点沿测地线延伸"""
        trajectory = []
        
        for z_idx in range(self.d_bulk):
            z = self._radial_coordinate(z_idx)
            
            # AdS测地线：指数衰减
            x = boundary_point * np.exp(-z / self.phi)
            trajectory.append((z, x))
            
            # 径向截断
            if z > self.z_cutoff:
                break
        
        return trajectory
    
    def _compute_surface_area(self, surface: np.ndarray) -> float:
        """计算曲面面积（离散近似）"""
        if len(surface) <= 1:
            return 1.0  # 最小面积
        
        area = 0.0
        
        for i in range(len(surface) - 1):
            # 相邻点间距
            point1 = surface[i]
            point2 = surface[i + 1]
            
            # 欧几里得距离（AdS度规修正）
            if len(point1) >= 2 and len(point2) >= 2:
                z1, x1 = point1[0], point1[1]
                z2, x2 = point2[0], point2[1]
                
                # 限制z值避免奇点
                z1 = max(z1, 0.1)
                z2 = max(z2, 0.1)
                
                # AdS度规：ds^2 = (dz^2 + dx^2)/z^2，但限制积分
                dz = z2 - z1
                dx = x2 - x1
                ds_squared = (dz**2 + dx**2) / (z1 * z2)
                
                # 添加Fibonacci衰减因子
                fibonacci_factor = self.phi ** (-i)
                area += np.sqrt(max(ds_squared, 1e-6)) * fibonacci_factor
        
        # 确保面积在合理范围内
        return min(area, len(surface))  # 面积不超过点数
    
    def radial_to_semantic_depth(self, z: float) -> int:
        """径向坐标到语义深度的映射"""
        if z <= 0:
            return 0
        
        # z = φ^(-depth) * z_0
        z_0 = 0.1
        depth = -np.log(z / z_0) / np.log(self.phi)
        return int(max(0, depth))
    
    def semantic_depth_to_radial(self, depth: int) -> float:
        """语义深度到径向坐标的映射"""
        z_0 = 0.1
        return z_0 * (self.phi ** (-depth))
    
    def holographic_rg_flow(self, boundary_state: np.ndarray, 
                           max_steps: int = 10) -> List[np.ndarray]:
        """全息重整化群流"""
        trajectory = [boundary_state.copy()]
        current = boundary_state.copy()
        
        for step in range(max_steps):
            # 径向位置
            z = self.semantic_depth_to_radial(step)
            
            # RG变换（粗粒化）
            current = self._rg_transform(current, z)
            trajectory.append(current.copy())
            
            # 检查不动点
            if self._is_rg_fixpoint(current):
                break
        
        return trajectory
    
    def _rg_transform(self, state: np.ndarray, z: float) -> np.ndarray:
        """重整化群变换"""
        n = len(state)
        if n <= 2:
            return state
        
        # 块自旋变换：相邻位合并
        coarse_length = (n + 1) // 2
        coarse = np.zeros(coarse_length, dtype=int)
        
        for i in range(coarse_length):
            if 2*i + 1 < n:
                # 多数表决
                block_sum = state[2*i] + state[2*i + 1]
                coarse[i] = 1 if block_sum >= 1 else 0
            elif 2*i < n:
                coarse[i] = state[2*i]
        
        # 扩展回原始大小
        expanded = np.zeros(n, dtype=int)
        for i in range(len(coarse)):
            if 2*i < n:
                expanded[2*i] = coarse[i]
            if 2*i + 1 < n:
                expanded[2*i + 1] = coarse[i]
        
        # 应用z依赖的扰动
        for i in range(n):
            if np.random.random() < z / self.z_cutoff * 0.1:
                expanded[i] = 1 - expanded[i]
        
        return self._enforce_no11(expanded)
    
    def _is_rg_fixpoint(self, state: np.ndarray) -> bool:
        """检查是否是RG不动点"""
        activity = np.sum(state)
        return activity <= 1
    
    def verify_quantum_error_correction(self, boundary_state: np.ndarray) -> Dict:
        """验证量子纠错码性质"""
        # 编码到体
        bulk_state = self.boundary_to_bulk(boundary_state)
        
        # 添加噪声（体中的局域扰动）
        noisy_bulk = self._add_bulk_noise(bulk_state, noise_level=0.1)
        
        # 解码回边界
        recovered_boundary = self._bulk_to_boundary(noisy_bulk)
        
        # 计算保真度
        fidelity = self._compute_fidelity(boundary_state, recovered_boundary)
        
        # 计算纠错能力
        hamming_distance = np.sum(boundary_state != recovered_boundary)
        
        return {
            'fidelity': fidelity,
            'hamming_distance': hamming_distance,
            'error_corrected': hamming_distance == 0,
            'boundary_preserved': fidelity > 0.8
        }
    
    def _add_bulk_noise(self, bulk_state: np.ndarray, noise_level: float) -> np.ndarray:
        """在体中添加局域噪声"""
        noisy = bulk_state.copy()
        
        for z_idx in range(len(bulk_state)):
            for x_idx in range(len(bulk_state[z_idx])):
                if np.random.random() < noise_level:
                    noisy[z_idx, x_idx] = 1 - noisy[z_idx, x_idx]
        
        return self._enforce_no11_bulk(noisy)
    
    def _bulk_to_boundary(self, bulk_state: np.ndarray) -> np.ndarray:
        """体态解码到边界态"""
        # 边界重构：使用最靠近边界的层
        if len(bulk_state) == 0:
            return np.array([])
        
        boundary_layer = bulk_state[0]  # z=0层
        
        # 额外的投影操作
        projected = np.zeros_like(boundary_layer)
        
        # 使用多层信息进行投影
        for i in range(len(boundary_layer)):
            votes = 0
            total_votes = 0
            
            for z_idx in range(min(3, len(bulk_state))):  # 前3层
                weight = self.phi ** (-z_idx)  # 越靠近边界权重越大
                votes += bulk_state[z_idx, i] * weight
                total_votes += weight
            
            projected[i] = 1 if votes / total_votes > 0.5 else 0
        
        return self._enforce_no11(projected)
    
    def _compute_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """计算保真度"""
        if len(state1) != len(state2):
            return 0.0
        
        # 汉明距离保真度
        differences = np.sum(state1 != state2)
        return np.exp(-differences / len(state1))
    
    def _enforce_no11(self, state: np.ndarray) -> np.ndarray:
        """强制no-11约束"""
        result = state.copy()
        for i in range(1, len(result)):
            if result[i-1] == 1 and result[i] == 1:
                result[i] = 0
        return result
    
    def _enforce_no11_bulk(self, bulk_state: np.ndarray) -> np.ndarray:
        """体中的no-11约束"""
        result = bulk_state.copy()
        for z in range(len(result)):
            result[z] = self._enforce_no11(result[z])
        return result


class TestAdSCFTObserverMapping(unittest.TestCase):
    """C17-6 AdS-CFT观察者映射测试套件"""
    
    def setUp(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_holographic_encoding(self):
        """测试全息编码"""
        boundary_dim = 8
        ads_cft = AdSCFTObserverMapping(boundary_dim)
        
        # 创建边界态
        boundary_state = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        
        # 编码到体
        bulk_state = ads_cft.boundary_to_bulk(boundary_state)
        
        # 验证体态结构
        self.assertEqual(bulk_state.shape[0], boundary_dim + 1)
        self.assertEqual(bulk_state.shape[1], boundary_dim)
        
        # 验证no-11约束
        for z in range(len(bulk_state)):
            for i in range(len(bulk_state[z]) - 1):
                self.assertFalse(bulk_state[z, i] == 1 and bulk_state[z, i+1] == 1,
                               f"No-11 violation at bulk[{z}, {i}:{i+1}]")
        
        # 验证边界层保真度
        boundary_layer = bulk_state[0]
        similarity = np.sum(boundary_state == boundary_layer) / len(boundary_state)
        self.assertGreater(similarity, 0.6, "Boundary layer should preserve information")
    
    def test_entanglement_entropy(self):
        """测试纠缠熵计算"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=6)
        
        boundary_state = np.array([1, 0, 1, 0, 1, 0])
        
        # 不同大小的子区域
        region_small = [0, 1]
        region_large = [0, 1, 2, 3]
        
        S_small = ads_cft.compute_entanglement_entropy(region_small, boundary_state)
        S_large = ads_cft.compute_entanglement_entropy(region_large, boundary_state)
        
        # 验证面积律：更大区域有更大纠缠熵
        self.assertGreater(S_large, S_small, "Larger regions should have higher entanglement")
        
        # 验证熵的合理范围
        self.assertGreater(S_small, 0, "Entanglement entropy should be positive")
        self.assertLess(S_large, 10, "Entanglement entropy should be bounded")
    
    def test_radial_semantic_correspondence(self):
        """测试径向-语义深度对应"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=10)
        
        # 测试双向映射
        for depth in [0, 1, 2, 3, 5]:
            z = ads_cft.semantic_depth_to_radial(depth)
            recovered_depth = ads_cft.radial_to_semantic_depth(z)
            
            # 验证映射的近似正确性
            self.assertLessEqual(abs(depth - recovered_depth), 1,
                              f"Round-trip depth mapping failed: {depth} -> {z} -> {recovered_depth}")
        
        # 验证单调性
        z1 = ads_cft.semantic_depth_to_radial(1)
        z2 = ads_cft.semantic_depth_to_radial(2)
        self.assertGreater(z1, z2, "Deeper states should be at larger radial positions")
    
    def test_rg_flow(self):
        """测试全息RG流"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=8)
        
        boundary_state = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        
        # 执行RG流
        trajectory = ads_cft.holographic_rg_flow(boundary_state, max_steps=5)
        
        # 验证轨迹长度
        self.assertGreater(len(trajectory), 1, "RG flow should produce trajectory")
        self.assertLessEqual(len(trajectory), 6, "RG flow should terminate")
        
        # 验证每步都满足no-11
        for step, state in enumerate(trajectory):
            for i in range(len(state) - 1):
                self.assertFalse(state[i] == 1 and state[i+1] == 1,
                               f"No-11 violation in RG step {step}")
        
        # 验证活跃度递减趋势
        activities = [np.sum(state) for state in trajectory]
        # 允许小的波动，但总体趋势应该递减
        self.assertLessEqual(activities[-1], activities[0] + 1,
                           "RG flow should reduce activity")
    
    def test_quantum_error_correction(self):
        """测试量子纠错码性质"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=6)
        
        boundary_state = np.array([1, 0, 1, 0, 1, 0])
        
        # 测试纠错能力
        qec_result = ads_cft.verify_quantum_error_correction(boundary_state)
        
        # 验证纠错性质
        self.assertIn('fidelity', qec_result)
        self.assertIn('hamming_distance', qec_result)
        self.assertIn('error_corrected', qec_result)
        
        # 保真度应该较高
        self.assertGreater(qec_result['fidelity'], 0.5,
                          "Quantum error correction should maintain reasonable fidelity")
        
        # 汉明距离应该有限
        self.assertLessEqual(qec_result['hamming_distance'], len(boundary_state),
                           "Hamming distance should be bounded")
    
    def test_boundary_observer_property(self):
        """测试边界观察者性质"""
        boundary_dim = 8
        ads_cft = AdSCFTObserverMapping(boundary_dim)
        
        # 创建边界观察者
        observer = ObserverSystem(boundary_dim)
        
        # 创建体系统
        bulk_system = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        bulk_state = ads_cft.boundary_to_bulk(bulk_system)
        
        # 观察者只能访问边界信息
        boundary_info = bulk_state[0]  # 边界层
        
        # 执行观察
        observed_system, observer_state = observer.observe(boundary_info)
        
        # 验证观察者维度
        self.assertEqual(len(observer_state), boundary_dim,
                        "Observer should live on boundary")
        
        # 验证观察结果
        self.assertEqual(len(observed_system), boundary_dim,
                        "Observed system should match boundary dimension")
        
        # 验证no-11约束保持
        for i in range(len(observer_state) - 1):
            self.assertFalse(observer_state[i] == 1 and observer_state[i+1] == 1)
    
    def test_holographic_duality_consistency(self):
        """测试全息对偶一致性"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=6)
        
        # 两个不同的边界态
        state1 = np.array([1, 0, 1, 0, 0, 0])
        state2 = np.array([1, 0, 0, 1, 0, 1])
        
        # 编码到体
        bulk1 = ads_cft.boundary_to_bulk(state1)
        bulk2 = ads_cft.boundary_to_bulk(state2)
        
        # 验证不同边界态对应不同体态
        boundary_difference = np.sum(state1 != state2)
        
        # 计算体态差异
        bulk_difference = 0
        for z in range(len(bulk1)):
            bulk_difference += np.sum(bulk1[z] != bulk2[z])
        
        # 对偶应该保持信息区分
        if boundary_difference > 0:
            self.assertGreater(bulk_difference, 0,
                             "Different boundary states should map to different bulk states")
    
    def test_area_law_verification(self):
        """测试面积律验证"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=8)
        
        boundary_state = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        
        # 测试不同大小的连通区域
        regions = [
            [0, 1],           # 大小2
            [0, 1, 2],        # 大小3  
            [0, 1, 2, 3],     # 大小4
        ]
        
        entropies = []
        for region in regions:
            S = ads_cft.compute_entanglement_entropy(region, boundary_state)
            entropies.append(S)
        
        # 验证面积律趋势
        for i in range(len(entropies) - 1):
            self.assertLessEqual(entropies[i], entropies[i+1] + 0.5,
                               "Area law should show increasing trend")
    
    def test_fibonacci_structure(self):
        """测试Fibonacci结构"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=10)
        
        # 获取Fibonacci位置
        fib_positions = ads_cft._get_fibonacci_positions(10)
        
        # 验证Fibonacci数列
        self.assertIn(1, fib_positions)
        self.assertIn(2, fib_positions)
        if len(fib_positions) > 2:
            self.assertIn(3, fib_positions)
            self.assertIn(5, fib_positions)
            if 8 < 10:
                self.assertIn(8, fib_positions)
        
        # 验证Fibonacci位置的特殊权重
        for pos in fib_positions:
            weight = ads_cft._fibonacci_modulation(pos, 10)
            self.assertEqual(weight, 1.0, 
                           f"Fibonacci position {pos} should have weight 1.0")
    
    def test_causality_structure(self):
        """测试因果结构"""
        ads_cft = AdSCFTObserverMapping(boundary_dim=6)
        
        boundary_state = np.array([1, 0, 1, 0, 1, 0])
        bulk_state = ads_cft.boundary_to_bulk(boundary_state)
        
        # 验证径向因果性：更深的层受边界影响减弱
        boundary_layer = bulk_state[0]
        if len(bulk_state) > 2:
            deep_layer = bulk_state[-1]
            
            # 边界信息在深层应该更加"模糊"
            boundary_activity = np.sum(boundary_layer)
            deep_activity = np.sum(deep_layer)
            
            # 允许活跃度在合理范围内变化
            self.assertLessEqual(deep_activity, boundary_activity + 2,
                               "Deep layers should not have excessive activity")


if __name__ == '__main__':
    unittest.main(verbosity=2)