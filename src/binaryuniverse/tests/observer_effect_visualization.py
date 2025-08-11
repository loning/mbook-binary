#!/usr/bin/env python3
"""
T14-2 观察者效应可视化
展示不同观察者如何测量到不同的物理常数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Tuple
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 黄金比率
phi = (1 + math.sqrt(5)) / 2

@dataclass
class ObserverType:
    """观察者类型"""
    name: str
    color: str
    psi_factor: float
    measured_alpha: float  # 1/α
    measured_sin2_theta: float
    position: Tuple[float, float]

def create_observer_network_diagram():
    """创建观察者网络图，展示不同观察者的测量值"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 定义不同类型的观察者
    observers = [
        ObserverType("Earth Human\n(Carbon-based)", "#4CAF50", 1.0, 137.036, 0.2312, (0, 0)),
        ObserverType("Mars Colony\n(Carbon-based)", "#8BC34A", 0.95, 144.2, 0.2298, (3, 1)),
        ObserverType("Silicon Being", "#2196F3", 1.1, 124.6, 0.2541, (-2, 2)),
        ObserverType("Plasma Entity", "#FF9800", 0.9, 152.3, 0.2089, (2, -2)),
        ObserverType("Quantum Observer", "#9C27B0", 0.7, 195.8, 0.1823, (-3, -1))
    ]
    
    # 左图：观察者网络和测量值
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    
    # 中心：ψ = ψ(ψ)递归结构
    center = Circle((0, 0), 0.5, color='black', alpha=0.3)
    ax1.add_patch(center)
    ax1.text(0, 0, 'ψ = ψ(ψ)', fontsize=12, ha='center', va='center', weight='bold')
    
    # 绘制观察者和测量值
    for obs in observers:
        # 观察者圆圈
        circle = Circle(obs.position, 0.8, color=obs.color, alpha=0.6)
        ax1.add_patch(circle)
        
        # 观察者标签
        ax1.text(obs.position[0], obs.position[1], obs.name, 
                fontsize=10, ha='center', va='center', weight='bold')
        
        # 测量值标注
        ax1.text(obs.position[0], obs.position[1] - 1.2, 
                f'1/α = {obs.measured_alpha:.1f}\nsin²θ_W = {obs.measured_sin2_theta:.4f}',
                fontsize=9, ha='center', va='top')
        
        # 连接到中心的线
        ax1.plot([0, obs.position[0]], [0, obs.position[1]], 
                'k--', alpha=0.3, linewidth=1)
    
    ax1.set_title('Observer-Dependent Measurements\n观察者依赖的测量值', fontsize=14, weight='bold')
    ax1.set_xlabel('Spatial Position', fontsize=12)
    ax1.set_ylabel('Energy Scale', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 右图：精细结构常数的观察者依赖性
    ax2.set_xlim(0.6, 1.2)
    ax2.set_ylim(100, 220)
    
    # 绘制理论曲线
    psi_range = np.linspace(0.6, 1.2, 100)
    alpha_theory = 143 / psi_range  # 基础值的观察者修正
    ax2.plot(psi_range, alpha_theory, 'k-', linewidth=2, 
            label='Theoretical Curve\n理论曲线')
    
    # 绘制观察者测量点
    for obs in observers:
        ax2.scatter(obs.psi_factor, obs.measured_alpha, 
                   color=obs.color, s=200, edgecolor='black', 
                   linewidth=2, zorder=5)
        ax2.text(obs.psi_factor + 0.02, obs.measured_alpha, 
                obs.name.split('\n')[0], fontsize=10)
    
    # 地球观察者特殊标注
    ax2.axhline(y=137.036, color='red', linestyle='--', alpha=0.5)
    ax2.text(1.15, 137.036, 'Earth Value\n地球值', 
            fontsize=10, va='center', color='red')
    
    ax2.set_xlabel('Observer ψ-Structure Factor\n观察者ψ结构因子', fontsize=12)
    ax2.set_ylabel('Measured 1/α\n测量的1/α值', fontsize=12)
    ax2.set_title('Fine Structure Constant vs Observer Structure\n精细结构常数与观察者结构', 
                 fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_coupling_hierarchy_diagram():
    """创建耦合常数层次图，展示递归深度与相互作用强度"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 递归层次
    levels = [
        {"name": "Strong Force\n强相互作用", "depth": 0, "coupling": 1.0, "color": "#FF5252"},
        {"name": "Electromagnetic\n电磁相互作用", "depth": 1, "coupling": 1/137, "color": "#4CAF50"},
        {"name": "Weak Force\n弱相互作用", "depth": 2, "coupling": 1/10000, "color": "#2196F3"},
        {"name": "Gravity (projected)\n引力(推测)", "depth": 3, "coupling": 1e-39, "color": "#9C27B0"}
    ]
    
    # 绘制递归层次
    for i, level in enumerate(levels):
        y_pos = 3 - i
        
        # 层次框
        width = 8 * (1 - i * 0.15)  # 递减的宽度
        rect = FancyBboxPatch((-width/2, y_pos - 0.4), width, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=level["color"], alpha=0.6,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # 标签
        ax.text(0, y_pos, level["name"], fontsize=14, 
               ha='center', va='center', weight='bold')
        
        # 耦合强度
        ax.text(-width/2 - 1, y_pos, f'g ≈ {level["coupling"]:.2e}',
               fontsize=12, ha='right', va='center')
        
        # 递归深度
        ax.text(width/2 + 1, y_pos, f'n = {level["depth"]}',
               fontsize=12, ha='left', va='center')
        
        # 观察者修正因子示意
        if i < 3:  # 不包括引力
            obs_factor = f'× ObserverFactor(ψ_obs)'
            ax.text(0, y_pos - 0.25, obs_factor, 
                   fontsize=10, ha='center', va='center', 
                   style='italic', alpha=0.7)
    
    # 递归箭头
    for i in range(len(levels) - 1):
        y1 = 3 - i - 0.5
        y2 = 3 - i - 1 + 0.5
        ax.annotate('', xy=(0, y2), xytext=(0, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax.text(0.5, (y1 + y2) / 2, 'ψ → ψ(ψ)', 
               fontsize=10, ha='left', va='center', style='italic')
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 4)
    ax.set_title('Coupling Hierarchy and Recursive Depth\n耦合层次与递归深度', 
                fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    # 添加说明文字
    explanation = (
        "The hierarchy of fundamental forces emerges from recursive depth.\n"
        "基本相互作用的层次源于递归深度。\n\n"
        "Measured values include observer-dependent corrections.\n"
        "测量值包含观察者依赖的修正。"
    )
    ax.text(0, -0.5, explanation, fontsize=11, ha='center', va='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
    
    return fig

def create_universal_principle_diagram():
    """创建普适原理图，展示所有观察者遵循同一递归原理"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 中心：普适递归原理
    center_box = FancyBboxPatch((-2, -0.5), 4, 1,
                               boxstyle="round,pad=0.2",
                               facecolor='gold', alpha=0.8,
                               edgecolor='black', linewidth=3)
    ax.add_patch(center_box)
    ax.text(0, 0, 'ψ = ψ(ψ)\nUniversal Principle\n普适原理', 
           fontsize=14, ha='center', va='center', weight='bold')
    
    # 不同观察者的投影
    projections = [
        {"angle": 0, "name": "Earth Observer\n地球观察者", "values": "α ≈ 1/137"},
        {"angle": 72, "name": "Silicon Observer\n硅基观察者", "values": "α ≈ 1/125"},
        {"angle": 144, "name": "Plasma Observer\n等离子观察者", "values": "α ≈ 1/152"},
        {"angle": 216, "name": "Quantum Observer\n量子观察者", "values": "α ≈ 1/196"},
        {"angle": 288, "name": "Crystal Observer\n晶体观察者", "values": "α ≈ 1/143"}
    ]
    
    radius = 4
    for proj in projections:
        angle_rad = np.radians(proj["angle"])
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        
        # 观察者框
        box = FancyBboxPatch((x - 1.5, y - 0.4), 3, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='lightblue', alpha=0.6,
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        
        # 标签和值
        ax.text(x, y + 0.1, proj["name"], fontsize=10, 
               ha='center', va='center', weight='bold')
        ax.text(x, y - 0.3, proj["values"], fontsize=9, 
               ha='center', va='center', style='italic')
        
        # 连接线
        ax.plot([0, x * 0.7], [0, y * 0.7], 'k--', alpha=0.5, linewidth=1)
        
        # 投影箭头
        ax.annotate('', xy=(x * 0.85, y * 0.85), xytext=(x * 0.75, y * 0.75),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    # 添加环形文字
    circle_text = "All observers see projections of the same reality • 所有观察者看到同一实在的不同投影"
    theta = np.linspace(0, 2*np.pi, len(circle_text))
    for i, char in enumerate(circle_text):
        x = 5.5 * np.cos(theta[i])
        y = 5.5 * np.sin(theta[i])
        angle = np.degrees(theta[i]) - 90
        if 90 < angle < 270:
            angle += 180
        ax.text(x, y, char, fontsize=8, ha='center', va='center',
               rotation=angle, alpha=0.7)
    
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Universal Recursion, Observer-Dependent Measurements\n'
                '普适递归，观察者依赖的测量', 
                fontsize=16, weight='bold', pad=20)
    
    return fig

def main():
    """生成所有可视化图表"""
    
    print("生成T14-2观察者效应可视化图表...")
    
    # 1. 观察者网络图
    fig1 = create_observer_network_diagram()
    fig1.savefig('observer_network_T14_2.png', dpi=300, bbox_inches='tight')
    print("✓ 观察者网络图已保存为 observer_network_T14_2.png")
    
    # 2. 耦合层次图
    fig2 = create_coupling_hierarchy_diagram()
    fig2.savefig('coupling_hierarchy_T14_2.png', dpi=300, bbox_inches='tight')
    print("✓ 耦合层次图已保存为 coupling_hierarchy_T14_2.png")
    
    # 3. 普适原理图
    fig3 = create_universal_principle_diagram()
    fig3.savefig('universal_principle_T14_2.png', dpi=300, bbox_inches='tight')
    print("✓ 普适原理图已保存为 universal_principle_T14_2.png")
    
    # 不显示图表，只保存
    # plt.show()
    
    print("\n所有T14-2可视化图表生成完成！")
    print("这些图表展示了：")
    print("1. 不同观察者测量到不同的物理常数")
    print("2. 递归深度与相互作用强度的关系")
    print("3. 所有观察者遵循同一个ψ = ψ(ψ)普适原理")

if __name__ == "__main__":
    main()