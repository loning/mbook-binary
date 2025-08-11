#!/usr/bin/env python3
"""
T17-2 φ-全息原理定理 - 可视化程序

可视化内容：
1. AdS/CFT对应关系网络
2. 全息纠缠熵的RT曲面
3. 黑洞信息演化与Page曲线
4. 全息复杂度增长
5. 熵增原理可视化
6. φ-量化约束分布
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Tuple, Dict
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phi_arithmetic import PhiReal
from no11_number_system import No11NumberSystem

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class T17_2_Visualizer:
    """T17-2 φ-全息原理可视化器"""
    
    def __init__(self):
        self.phi = 1.618033988749895
        self.no11 = No11NumberSystem()
        self.fig_size = (15, 12)
        
        # 创建输出目录
        self.output_dir = "T17_2_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_all(self):
        """生成所有可视化"""
        print("生成T17-2 φ-全息原理定理可视化...")
        
        # 1. AdS/CFT对应关系网络
        self.plot_ads_cft_correspondence()
        
        # 2. 全息纠缠熵
        self.plot_holographic_entanglement()
        
        # 3. 黑洞演化与Page曲线
        self.plot_black_hole_evolution()
        
        # 4. 全息复杂度
        self.plot_holographic_complexity()
        
        # 5. 熵增原理
        self.plot_entropy_increase()
        
        # 6. φ-量化约束
        self.plot_phi_quantization()
        
        print(f"所有可视化已保存到 {self.output_dir}/ 目录")
    
    def plot_ads_cft_correspondence(self):
        """可视化AdS/CFT对应关系"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：AdS空间结构
        ax1.set_title("AdS₃ 空间结构", fontsize=14, fontweight='bold')
        
        # 绘制AdS边界
        theta = np.linspace(0, 2*np.pi, 100)
        boundary_r = 1.0
        ax1.plot(boundary_r * np.cos(theta), boundary_r * np.sin(theta), 
                'r-', linewidth=3, label='AdS边界')
        
        # 绘制体积切片
        for r in [0.2, 0.4, 0.6, 0.8]:
            ax1.plot(r * np.cos(theta), r * np.sin(theta), 
                    'b--', alpha=0.5, linewidth=1)
        
        # 标记特殊点
        ads_points = [(0.8, 0), (0.8*np.cos(np.pi/3), 0.8*np.sin(np.pi/3)), 
                     (0.8*np.cos(2*np.pi/3), 0.8*np.sin(2*np.pi/3))]
        for i, (x, y) in enumerate(ads_points):
            ax1.plot(x, y, 'go', markersize=8)
            ax1.annotate(f'AdS场_{i+1}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：CFT边界理论
        ax2.set_title("CFT₂ 边界理论", fontsize=14, fontweight='bold')
        
        # 绘制CFT边界线
        x_boundary = np.linspace(-1, 1, 100)
        ax2.plot(x_boundary, np.ones_like(x_boundary), 'r-', linewidth=3, 
                label='CFT边界')
        
        # 标记算子位置
        cft_positions = [-0.6, 0, 0.6]
        scaling_dims = [self.phi, self.phi**2, 2*self.phi]
        
        for i, (pos, dim) in enumerate(zip(cft_positions, scaling_dims)):
            # 算子高度正比于标度维度
            height = dim / max(scaling_dims) * 0.8
            ax2.plot([pos, pos], [1, 1-height], 'b-', linewidth=3)
            ax2.plot(pos, 1-height, 'bo', markersize=10)
            ax2.annotate(f'O_{i+1}(Δ={dim:.2f})', 
                        (pos, 1-height), xytext=(0, -20), 
                        textcoords='offset points', ha='center', fontsize=10)
        
        # 绘制对应关系箭头
        ax2.annotate('', xy=(0.6, 0.3), xytext=(-0.6, 0.3),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax2.text(0, 0.25, 'AdS/CFT\n对应关系', ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(0, 1.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ads_cft_correspondence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_holographic_entanglement(self):
        """可视化全息纠缠熵"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：RT曲面
        ax1.set_title("Ryu-Takayanagi 最小曲面", fontsize=14, fontweight='bold')
        
        # 绘制AdS空间（Poincaré盘表示）
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, alpha=0.3)
        
        # 绘制边界区域A
        region_A = np.linspace(-0.8, -0.2, 50)
        ax1.plot(region_A, np.sqrt(1 - region_A**2), 'r-', linewidth=4, label='边界区域A')
        
        # 绘制最小曲面（测地线）
        t = np.linspace(0, 1, 100)
        # 连接区域A端点的测地线
        x_geodesic = -0.8 + 0.6 * t
        y_geodesic = np.sqrt(1 - x_geodesic**2) * (1 - 2*t*(1-t))
        ax1.plot(x_geodesic, y_geodesic, 'b-', linewidth=3, label='RT最小曲面')
        
        # 标记关键点
        ax1.plot([-0.8, -0.2], [np.sqrt(1-0.64), np.sqrt(1-0.04)], 'ro', markersize=8)
        ax1.plot([x_geodesic[50]], [y_geodesic[50]], 'bo', markersize=8)
        
        # 添加熵公式
        ax1.text(0.3, 0.7, r'$S_A = \frac{\text{Area}(\gamma_A)}{4G_N}$', 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：纠缠熵随区域大小变化
        ax2.set_title("纠缠熵随区域大小变化", fontsize=14, fontweight='bold')
        
        # 计算不同区域大小的纠缠熵
        region_sizes = np.linspace(0.1, 1.5, 50)
        entanglement_entropies = []
        
        for size in region_sizes:
            # 简化的RT公式：S ~ log(size) + φ修正
            entropy = np.log(size + 0.1) + self.phi * size / 10
            entanglement_entropies.append(max(0, entropy))
        
        ax2.plot(region_sizes, entanglement_entropies, 'b-', linewidth=3, 
                label='纠缠熵 $S_A$')
        
        # 标记φ-量化点
        phi_points = [0.618, 1.0, 1.618]
        for point in phi_points:
            if point <= max(region_sizes):
                idx = np.argmin(np.abs(region_sizes - point))
                ax2.plot(point, entanglement_entropies[idx], 'ro', markersize=8)
                ax2.axvline(x=point, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('区域大小')
        ax2.set_ylabel('纠缠熵')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/holographic_entanglement.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_black_hole_evolution(self):
        """可视化黑洞演化与Page曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 生成演化数据
        time_steps = np.linspace(0, 10, 100)
        initial_mass = 10.0
        masses = initial_mass * np.exp(-time_steps / 5)  # 指数衰减
        
        # 计算对应的熵
        black_hole_entropies = 4 * np.pi * masses**2  # S_BH ~ M²
        radiation_entropies = time_steps * 0.5  # 辐射熵线性增长
        
        # Page曲线：取最小值
        page_curve = np.minimum(radiation_entropies, 
                               black_hole_entropies[0] - black_hole_entropies + radiation_entropies)
        
        # 图1：黑洞质量演化
        ax1.set_title("黑洞质量演化", fontsize=14, fontweight='bold')
        ax1.plot(time_steps, masses, 'b-', linewidth=3, label='黑洞质量 M(t)')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('时间')
        ax1.set_ylabel('质量')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 图2：熵演化
        ax2.set_title("熵演化", fontsize=14, fontweight='bold')
        ax2.plot(time_steps, black_hole_entropies, 'r-', linewidth=3, label='黑洞熵 $S_{BH}$')
        ax2.plot(time_steps, radiation_entropies, 'g-', linewidth=3, label='辐射熵 $S_{rad}$')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('熵')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 图3：Page曲线
        ax3.set_title("Page曲线", fontsize=14, fontweight='bold')
        ax3.plot(time_steps, page_curve, 'purple', linewidth=4, label='Page曲线')
        ax3.fill_between(time_steps, 0, page_curve, alpha=0.3, color='purple')
        
        # 标记Page时间（熵开始下降的时间点）
        page_time_idx = np.argmax(page_curve)
        page_time = time_steps[page_time_idx]
        ax3.axvline(x=page_time, color='red', linestyle='--', linewidth=2)
        ax3.annotate(f'Page时间 t≈{page_time:.1f}', 
                    xy=(page_time, page_curve[page_time_idx]), 
                    xytext=(page_time+1, page_curve[page_time_idx]+50),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        ax3.set_xlabel('时间')
        ax3.set_ylabel('von Neumann熵')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 图4：信息悖论解决
        ax4.set_title("信息悖论的φ-全息解决", fontsize=14, fontweight='bold')
        
        # 绘制熵增原理曲线
        entropy_increase = np.cumsum(np.ones_like(time_steps) * 0.01)  # 严格递增
        ax4.plot(time_steps, entropy_increase, 'orange', linewidth=3, 
                label='总熵增 (唯一公理)')
        
        # 标记φ-量化点
        phi_times = [1/self.phi, 1.0, self.phi, self.phi**2]
        for t in phi_times:
            if t <= max(time_steps):
                idx = np.argmin(np.abs(time_steps - t))
                ax4.plot(t, entropy_increase[idx], 'ro', markersize=8)
        
        ax4.text(5, 0.3, '根据唯一公理：\n自指完备系统\n必然熵增', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'),
                fontsize=11)
        
        ax4.set_xlabel('时间')
        ax4.set_ylabel('总熵增')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/black_hole_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_holographic_complexity(self):
        """可视化全息复杂度"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：体积复杂度
        ax1.set_title("全息复杂度：体积提案", fontsize=14, fontweight='bold')
        
        # 创建AdS空间的体积结构
        x = np.linspace(-1, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # 体积复杂度函数（随深度增加）
        volume_complexity = (1 - Y) * np.exp(-(X**2) / 0.5)
        
        im1 = ax1.contourf(X, Y, volume_complexity, levels=20, cmap='viridis')
        ax1.contour(X, Y, volume_complexity, levels=10, colors='white', alpha=0.5)
        
        # 标记WDW patch边界
        wdw_boundary_x = np.linspace(-0.8, 0.8, 100)
        wdw_boundary_y = 0.2 + 0.6 * (1 - wdw_boundary_x**2)
        ax1.plot(wdw_boundary_x, wdw_boundary_y, 'r-', linewidth=3, 
                label='WDW边界')
        
        # 标记边界状态
        ax1.plot([-0.5, 0, 0.5], [1, 1, 1], 'ro', markersize=10, 
                label='边界状态')
        
        ax1.set_xlabel('边界坐标 x')
        ax1.set_ylabel('AdS径向坐标 z')
        ax1.legend()
        plt.colorbar(im1, ax=ax1, label='体积复杂度')
        
        # 右图：作用量复杂度
        ax2.set_title("全息复杂度：作用量提案", fontsize=14, fontweight='bold')
        
        # 时间演化的复杂度
        times = np.linspace(0, 10, 100)
        
        # 体积复杂度：线性增长至平台
        volume_C = np.minimum(times * self.phi, self.phi**2 * np.ones_like(times))
        
        # 作用量复杂度：线性增长
        action_C = times * (2 * self.phi)
        
        ax2.plot(times, volume_C, 'b-', linewidth=3, label='体积复杂度 $C_V$')
        ax2.plot(times, action_C, 'r-', linewidth=3, label='作用量复杂度 $C_A$')
        
        # 标记复杂度平台
        plateau_time = self.phi
        ax2.axvline(x=plateau_time, color='blue', linestyle='--', alpha=0.7)
        ax2.annotate(f'复杂度平台\nt={plateau_time:.2f}', 
                    xy=(plateau_time, self.phi**2), 
                    xytext=(plateau_time+1, self.phi**2+1),
                    arrowprops=dict(arrowstyle='->', color='blue'))
        
        # 标记φ-量化时间点
        phi_times = [1/self.phi, 1.0, self.phi, self.phi**2]
        for t in phi_times:
            if t <= max(times):
                ax2.axvline(x=t, color='orange', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('时间')
        ax2.set_ylabel('复杂度')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/holographic_complexity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_entropy_increase(self):
        """可视化熵增原理"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 图1：编码过程的熵增
        ax1.set_title("全息编码的熵增", fontsize=14, fontweight='bold')
        
        encoding_steps = np.arange(1, 11)
        base_entropy = 0.1 * encoding_steps
        complexity_entropy = np.log(1 + encoding_steps)
        self_ref_entropy = encoding_steps * 0.05
        total_entropy = base_entropy + complexity_entropy + self_ref_entropy
        
        ax1.bar(encoding_steps, base_entropy, label='基本熵增', alpha=0.7)
        ax1.bar(encoding_steps, complexity_entropy, bottom=base_entropy, 
               label='编码复杂度', alpha=0.7)
        ax1.bar(encoding_steps, self_ref_entropy, 
               bottom=base_entropy+complexity_entropy, 
               label='自指性熵增', alpha=0.7)
        
        ax1.plot(encoding_steps, total_entropy, 'r-', linewidth=3, 
                marker='o', markersize=6, label='总熵增')
        
        ax1.set_xlabel('编码步骤')
        ax1.set_ylabel('熵增')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：自指循环与熵增
        ax2.set_title("自指循环必然熵增", fontsize=14, fontweight='bold')
        
        # 绘制自指循环
        angles = np.linspace(0, 2*np.pi, 100)
        radius = 1
        x_circle = radius * np.cos(angles)
        y_circle = radius * np.sin(angles)
        
        ax2.plot(x_circle, y_circle, 'b-', linewidth=3)
        
        # 添加箭头表示循环方向
        arrow_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        for angle in arrow_angles:
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            dx, dy = -0.3 * np.sin(angle), 0.3 * np.cos(angle)
            ax2.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue')
        
        # 标记自指点
        ax2.plot(0, 0, 'ro', markersize=15, label='ψ = ψ(ψ)')
        ax2.text(0, 0, 'ψ(ψ)', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 熵增螺旋
        spiral_r = np.linspace(1, 1.5, 50)
        spiral_angles = np.linspace(0, 4*np.pi, 50)
        spiral_x = spiral_r * np.cos(spiral_angles)
        spiral_y = spiral_r * np.sin(spiral_angles)
        
        ax2.plot(spiral_x, spiral_y, 'r--', linewidth=2, alpha=0.7, 
                label='熵增轨迹')
        
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3：φ-量化与熵增
        ax3.set_title("φ-量化约束下的熵增", fontsize=14, fontweight='bold')
        
        # φ的幂次
        phi_powers = np.array([self.phi**n for n in range(-3, 4)])
        phi_entropies = np.array([abs(n) * 0.1 + 0.05 for n in range(-3, 4)])
        
        bars = ax3.bar(range(len(phi_powers)), phi_entropies, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(phi_powers))))
        
        # 添加数值标签
        for i, (power, entropy) in enumerate(zip(phi_powers, phi_entropies)):
            ax3.text(i, entropy + 0.01, f'φ^{i-3}\n≈{power:.2f}', 
                    ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('φ的幂次')
        ax3.set_ylabel('对应熵增')
        ax3.set_xticks(range(len(phi_powers)))
        ax3.set_xticklabels([f'φ^{i-3}' for i in range(len(phi_powers))])
        ax3.grid(True, alpha=0.3)
        
        # 图4：no-11约束与熵
        ax4.set_title("no-11约束对熵的影响", fontsize=14, fontweight='bold')
        
        # 生成一些数字及其no-11约束检查
        numbers = np.arange(1, 21)
        valid_numbers = []
        entropies = []
        
        for n in numbers:
            # 简化的no-11检查（避免连续的1）
            binary = bin(n)[2:]
            is_valid = '11' not in binary
            valid_numbers.append(is_valid)
            
            # 熵与数字复杂度相关
            entropy = len(binary) * 0.1 + (0.05 if is_valid else 0.02)
            entropies.append(entropy)
        
        colors = ['green' if valid else 'red' for valid in valid_numbers]
        bars = ax4.bar(numbers, entropies, color=colors, alpha=0.7)
        
        # 添加图例
        valid_patch = patches.Patch(color='green', label='no-11有效')
        invalid_patch = patches.Patch(color='red', label='no-11无效')
        ax4.legend(handles=[valid_patch, invalid_patch])
        
        ax4.set_xlabel('数字')
        ax4.set_ylabel('编码熵')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/entropy_increase.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_phi_quantization(self):
        """可视化φ-量化约束"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 图1：φ数列的分布
        ax1.set_title("φ-量化数值分布", fontsize=14, fontweight='bold')
        
        # 生成φ相关数值
        phi_values = []
        labels = []
        
        # φ的幂次
        for n in range(-3, 4):
            value = self.phi ** n
            phi_values.append(value)
            labels.append(f'φ^{n}' if n != 0 else '1')
        
        # φ的组合
        combinations = [
            (self.phi - 1, 'φ-1'),
            (self.phi + 1, 'φ+1'),
            (2 * self.phi, '2φ'),
            (self.phi / 2, 'φ/2'),
            (self.phi**2 - self.phi, 'φ²-φ')
        ]
        
        for value, label in combinations:
            phi_values.append(value)
            labels.append(label)
        
        # 排序
        sorted_indices = np.argsort(phi_values)
        phi_values = [phi_values[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        # 绘制分布
        y_pos = np.arange(len(phi_values))
        bars = ax1.barh(y_pos, phi_values, color=plt.cm.plasma(np.linspace(0, 1, len(phi_values))))
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('数值')
        ax1.grid(True, alpha=0.3)
        
        # 添加φ≈1.618的标记线
        ax1.axvline(x=self.phi, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(self.phi + 0.1, len(phi_values)/2, f'φ≈{self.phi:.3f}', 
                rotation=90, va='center', fontsize=10)
        
        # 图2：φ-量化检测函数
        ax2.set_title("φ-量化检测函数", fontsize=14, fontweight='bold')
        
        test_values = np.linspace(0, 5, 1000)
        quantization_scores = []
        
        for val in test_values:
            # 计算与φ相关值的最小距离
            phi_related = [1.0, self.phi, self.phi**2, 1/self.phi, 
                          self.phi-1, self.phi+1, 2*self.phi, self.phi/2]
            min_distance = min(abs(val - phi_val) for phi_val in phi_related)
            
            # 量化得分：距离越小得分越高
            score = np.exp(-min_distance * 5)
            quantization_scores.append(score)
        
        ax2.plot(test_values, quantization_scores, 'b-', linewidth=2)
        ax2.fill_between(test_values, 0, quantization_scores, alpha=0.3)
        
        # 标记主要φ值
        main_phi_values = [1/self.phi, 1.0, self.phi, self.phi**2, 2*self.phi]
        for val in main_phi_values:
            if val <= max(test_values):
                ax2.axvline(x=val, color='red', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('测试数值')
        ax2.set_ylabel('φ-量化得分')
        ax2.grid(True, alpha=0.3)
        
        # 图3：Zeckendorf表示
        ax3.set_title("Zeckendorf表示 (Fibonacci基)", fontsize=14, fontweight='bold')
        
        # 生成Fibonacci数列
        fib = [1, 1]
        while fib[-1] < 100:
            fib.append(fib[-1] + fib[-2])
        fib = fib[2:]  # 去掉重复的1
        
        # 选择一些数字进行Zeckendorf分解
        test_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        for i, num in enumerate(test_numbers):
            zeck_rep = self._zeckendorf_representation(num, fib)
            
            # 绘制分解
            y = i
            x_start = 0
            for j, (fib_num, used) in enumerate(zip(fib[:len(zeck_rep)], zeck_rep)):
                width = fib_num
                color = 'gold' if used else 'lightgray'
                rect = patches.Rectangle((x_start, y-0.4), width, 0.8, 
                                       facecolor=color, edgecolor='black')
                ax3.add_patch(rect)
                
                if used:
                    ax3.text(x_start + width/2, y, str(fib_num), 
                            ha='center', va='center', fontweight='bold')
                
                x_start += width
            
            # 标记总数
            ax3.text(-5, y, f'{num} =', ha='right', va='center', fontsize=10)
        
        ax3.set_xlim(-10, 100)
        ax3.set_ylim(-0.5, len(test_numbers)-0.5)
        ax3.set_ylabel('测试数字')
        ax3.set_xlabel('Fibonacci分解')
        ax3.set_yticks(range(len(test_numbers)))
        ax3.set_yticklabels(test_numbers)
        
        # 图4：no-11约束可视化
        ax4.set_title("no-11约束可视化", fontsize=14, fontweight='bold')
        
        # 生成二进制表示
        numbers = list(range(1, 32))
        valid_invalid = []
        
        for num in numbers:
            binary = bin(num)[2:]
            has_11 = '11' in binary
            valid_invalid.append(not has_11)
        
        # 创建颜色映射
        colors = ['green' if valid else 'red' for valid in valid_invalid]
        
        # 绘制散点图
        x_coords = [i % 8 for i in range(len(numbers))]
        y_coords = [i // 8 for i in range(len(numbers))]
        
        scatter = ax4.scatter(x_coords, y_coords, c=colors, s=200, alpha=0.7)
        
        # 添加数字标签
        for i, (x, y, num) in enumerate(zip(x_coords, y_coords, numbers)):
            binary = bin(num)[2:]
            ax4.text(x, y, f'{num}\n{binary}', ha='center', va='center', 
                    fontsize=8, fontweight='bold')
        
        # 添加图例和标注
        valid_patch = patches.Patch(color='green', label='无连续11 (有效)')
        invalid_patch = patches.Patch(color='red', label='有连续11 (无效)')
        ax4.legend(handles=[valid_patch, invalid_patch])
        
        ax4.set_xlim(-0.5, 7.5)
        ax4.set_ylim(-0.5, 4.5)
        ax4.set_title("no-11约束：二进制中无连续11")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phi_quantization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _zeckendorf_representation(self, n: int, fib_sequence: List[int]) -> List[bool]:
        """计算数字的Zeckendorf表示"""
        if n == 0:
            return [False] * len(fib_sequence)
        
        result = [False] * len(fib_sequence)
        remaining = n
        
        # 从最大的Fibonacci数开始
        for i in range(len(fib_sequence)-1, -1, -1):
            if fib_sequence[i] <= remaining:
                result[i] = True
                remaining -= fib_sequence[i]
        
        return result

def main():
    """主函数"""
    visualizer = T17_2_Visualizer()
    visualizer.visualize_all()

if __name__ == "__main__":
    main()