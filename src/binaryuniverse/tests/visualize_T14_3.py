#!/usr/bin/env python3
"""
T14-3 φ-超对称与弦理论可视化
生成展示超对称、弦态、D-膜和紧致化的图表
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = {
    'boson': '#2E86AB',      # 蓝色 - 玻色子
    'fermion': '#A23B72',    # 紫色 - 费米子
    'string': '#F18F01',     # 橙色 - 弦
    'brane': '#C73E1D',      # 红色 - D-膜
    'compact': '#6A994E',    # 绿色 - 紧致化
    'landscape': '#BC4B51',  # 深红 - 景观
    'no11': '#F4E285',       # 黄色 - no-11约束
    'background': '#F8F8F8'
}

def create_supersymmetry_diagram():
    """创建超对称变换图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 标题
    ax.text(0.5, 0.95, 'φ-Supersymmetry: Recursive Symmetry', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.90, 'ψ = ψ(ψ) Manifests as Boson-Fermion Duality', 
            ha='center', va='top', fontsize=12, style='italic')
    
    # 玻色子态
    boson_box = FancyBboxPatch((0.1, 0.6), 0.3, 0.15,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['boson'],
                                edgecolor='black',
                                alpha=0.8)
    ax.add_patch(boson_box)
    ax.text(0.25, 0.675, 'Boson State', ha='center', va='center', 
            fontsize=12, color='white', fontweight='bold')
    ax.text(0.25, 0.63, 'ψ(ψ(ψ))', ha='center', va='center', 
            fontsize=10, color='white')
    ax.text(0.25, 0.58, 'Even Recursion', ha='center', va='center', 
            fontsize=9, color='white', style='italic')
    
    # 费米子态
    fermion_box = FancyBboxPatch((0.6, 0.6), 0.3, 0.15,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS['fermion'],
                                 edgecolor='black',
                                 alpha=0.8)
    ax.add_patch(fermion_box)
    ax.text(0.75, 0.675, 'Fermion State', ha='center', va='center', 
            fontsize=12, color='white', fontweight='bold')
    ax.text(0.75, 0.63, 'ψ(ψ)', ha='center', va='center', 
            fontsize=10, color='white')
    ax.text(0.75, 0.58, 'Odd Recursion', ha='center', va='center', 
            fontsize=9, color='white', style='italic')
    
    # 超对称变换
    arrow1 = patches.FancyArrowPatch((0.4, 0.675), (0.6, 0.675),
                                     connectionstyle="arc3,rad=0.3",
                                     arrowstyle='->',
                                     mutation_scale=20,
                                     linewidth=2,
                                     color='black')
    ax.add_patch(arrow1)
    ax.text(0.5, 0.74, 'Q', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    arrow2 = patches.FancyArrowPatch((0.6, 0.625), (0.4, 0.625),
                                     connectionstyle="arc3,rad=-0.3",
                                     arrowstyle='->',
                                     mutation_scale=20,
                                     linewidth=2,
                                     color='black')
    ax.add_patch(arrow2)
    ax.text(0.5, 0.56, 'Q†', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # 超代数关系
    ax.text(0.5, 0.45, 'Superalgebra Relations:', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(0.5, 0.40, '{Q, Q†} = 2H^φ', ha='center', va='center', 
            fontsize=11, family='monospace')
    ax.text(0.5, 0.36, 'Q² = 0', ha='center', va='center', 
            fontsize=11, family='monospace')
    
    # 递归深度示意
    ax.text(0.1, 0.25, 'Recursion Depth:', ha='left', va='top', 
            fontsize=11, fontweight='bold')
    
    # 递归层级
    levels = ['ψ', 'ψ(ψ)', 'ψ(ψ(ψ))', 'ψ(ψ(ψ(ψ)))']
    types = ['—', 'Fermion', 'Boson', 'Fermion']
    colors = ['gray', COLORS['fermion'], COLORS['boson'], COLORS['fermion']]
    
    for i, (level, ptype, color) in enumerate(zip(levels, types, colors)):
        y = 0.20 - i * 0.04
        ax.text(0.15, y, f'n={i}:', ha='left', va='center', fontsize=10)
        ax.text(0.25, y, level, ha='left', va='center', fontsize=10, 
                family='monospace')
        ax.text(0.45, y, ptype, ha='left', va='center', fontsize=10,
                color=color, fontweight='bold')
    
    # no-11约束标注
    constraint_box = FancyBboxPatch((0.65, 0.08), 0.3, 0.15,
                                    boxstyle="round,pad=0.02",
                                    facecolor=COLORS['no11'],
                                    edgecolor='black',
                                    alpha=0.6)
    ax.add_patch(constraint_box)
    ax.text(0.8, 0.155, 'no-11 Constraint', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(0.8, 0.12, 'Preserves SUSY', ha='center', va='center', 
            fontsize=9)
    ax.text(0.8, 0.09, 'in φ-encoding', ha='center', va='center', 
            fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('supersymmetry_T14_3.png', dpi=300, bbox_inches='tight', 
                facecolor=COLORS['background'])
    plt.close()

def create_string_modes_diagram():
    """创建弦振动模式图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # 整体标题
    fig.suptitle('φ-String Theory: Vibrational Modes with no-11 Constraint', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 左图：弦的振动模式
    ax1.set_title('String Oscillation Modes', fontsize=14)
    
    # 绘制弦
    x = np.linspace(0, 2*np.pi, 1000)
    
    # 基态
    y0 = np.zeros_like(x)
    ax1.plot(x, y0 + 4, 'k-', linewidth=2, label='n=0 (ground)')
    
    # 激发态（满足no-11约束）
    valid_modes = [1, 3, 5, 8]  # Fibonacci数，避免连续
    for i, n in enumerate(valid_modes):
        y = 0.3 * np.sin(n * x)
        ax1.plot(x, y + 3 - i*0.8, color=COLORS['string'], 
                linewidth=2, alpha=0.8, label=f'n={n}')
        ax1.text(-0.5, 3 - i*0.8, f'F_{n}', ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # 禁止的模式（虚线）
    forbidden_modes = [2, 4]  # 会导致连续Fibonacci
    for n in forbidden_modes:
        y = 0.3 * np.sin(n * x)
        ax1.plot(x, y - 1.5, '--', color='gray', 
                linewidth=1, alpha=0.5, label=f'n={n} (forbidden)')
    
    ax1.set_xlabel('σ (String Parameter)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_xlim(-1, 2*np.pi + 1)
    ax1.set_ylim(-2.5, 4.5)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图：质量谱
    ax2.set_title('Mass Spectrum with no-11 Constraint', fontsize=14)
    
    # 允许的质量级
    allowed_masses = []
    forbidden_masses = []
    
    # 生成质量谱
    for n in range(10):
        mass = np.sqrt(n)
        if n in [0, 1, 3, 5, 8]:  # 满足no-11的模式
            allowed_masses.append((n, mass))
        else:
            forbidden_masses.append((n, mass))
    
    # 绘制谱线
    for n, m in allowed_masses:
        ax2.vlines(n, 0, m, colors=COLORS['string'], linewidth=3, alpha=0.8)
        ax2.text(n, m + 0.1, f'M²={n}', ha='center', va='bottom', fontsize=9)
    
    for n, m in forbidden_masses:
        ax2.vlines(n, 0, m, colors='gray', linewidth=1, 
                  linestyles='dashed', alpha=0.5)
    
    ax2.set_xlabel('n (Level Number)', fontsize=12)
    ax2.set_ylabel('Mass M/M_string', fontsize=12)
    ax2.set_xlim(-0.5, 9.5)
    ax2.set_ylim(0, 3.5)
    ax2.grid(True, alpha=0.3)
    
    # 标注
    ax2.text(7, 2.8, 'Allowed states', color=COLORS['string'], 
            fontsize=10, fontweight='bold')
    ax2.text(7, 2.5, 'Forbidden by no-11', color='gray', 
            fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('string_modes_T14_3.png', dpi=300, bbox_inches='tight',
                facecolor=COLORS['background'])
    plt.close()

def create_dbrane_compactification_diagram():
    """创建D-膜和紧致化图"""
    fig = plt.figure(figsize=(14, 10))
    
    # 创建子图布局
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # 整体标题
    fig.suptitle('D-branes and Compactification in φ-String Theory', 
                 fontsize=16, fontweight='bold')
    
    # 子图1：D-膜堆叠
    ax1.set_title('D-brane Stack', fontsize=14)
    
    # 绘制D-膜
    for i in range(4):
        y = i * 0.3
        # 膜
        rect = FancyBboxPatch((0.1, y), 0.8, 0.2,
                             boxstyle="round,pad=0.01",
                             facecolor=COLORS['brane'],
                             edgecolor='black',
                             alpha=0.7 - i*0.1)
        ax1.add_patch(rect)
        ax1.text(0.5, y + 0.1, f'D{3-i}-brane', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
        
        # 开弦
        if i < 3:
            for j in range(3):
                x1 = 0.2 + j * 0.3
                x2 = x1 + np.random.uniform(-0.1, 0.1)
                y1 = y + 0.2
                y2 = y + 0.3
                ax1.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.5)
    
    ax1.text(0.5, -0.15, 'Open strings end on D-branes', 
            ha='center', va='center', fontsize=10, style='italic')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.2, 1.3)
    ax1.axis('off')
    
    # 子图2：膜张力与φ关系
    ax2.set_title('D-brane Tension', fontsize=14)
    
    p_values = np.arange(0, 10)
    phi = (1 + np.sqrt(5)) / 2
    
    # 张力公式：T_Dp ~ φ^f(p)
    tensions = []
    for p in p_values:
        if p <= 3:
            factor = phi
        elif p <= 6:
            factor = phi ** 0.5
        else:
            factor = phi ** (-0.5)
        tension = factor / (2 * np.pi) ** p
        tensions.append(tension)
    
    ax2.semilogy(p_values, tensions, 'o-', color=COLORS['brane'], 
                linewidth=2, markersize=8)
    ax2.set_xlabel('p (spatial dimensions)', fontsize=12)
    ax2.set_ylabel('T_Dp (φ-corrected)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 9.5)
    
    # 标注no-11修正
    ax2.text(7, tensions[2], 'φ-factors from\nno-11 constraint', 
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=COLORS['no11'], alpha=0.5))
    
    # 子图3：紧致化
    ax3.set_title('Calabi-Yau Compactification with Zeckendorf Structure', 
                  fontsize=14)
    
    # 绘制高维空间
    ax3.text(0.05, 0.8, '10D Spacetime', fontsize=12, fontweight='bold')
    
    # 大维度
    large_dims = FancyBboxPatch((0.1, 0.6), 0.35, 0.15,
                               boxstyle="round,pad=0.02",
                               facecolor='lightblue',
                               edgecolor='black')
    ax3.add_patch(large_dims)
    ax3.text(0.275, 0.675, '4D Minkowski', ha='center', va='center',
            fontsize=11)
    
    # 紧致维度
    compact_dims = []
    fib_indices = [1, 2, 3, 5, 8]  # Fibonacci indices
    
    for i, f in enumerate(fib_indices[:3]):
        x = 0.55 + i * 0.12
        y = 0.65
        circle = Circle((x, y), 0.04, facecolor=COLORS['compact'],
                       edgecolor='black', alpha=0.8)
        ax3.add_patch(circle)
        ax3.text(x, y - 0.08, f'R_{f}', ha='center', va='center',
                fontsize=9)
    
    ax3.text(0.67, 0.75, '6D Calabi-Yau', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # 箭头
    arrow = patches.FancyArrowPatch((0.45, 0.675), (0.52, 0.675),
                                   arrowstyle='->',
                                   mutation_scale=20,
                                   linewidth=2)
    ax3.add_patch(arrow)
    
    # 体积公式
    ax3.text(0.5, 0.45, 'Compactification Volume:', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.38, r'$V_{CY}^φ = V_0 \prod_{i \in \mathrm{ValidSet}} (1 + \epsilon_i φ^{F_i})$',
            ha='center', va='center', fontsize=11)
    
    # Kaluza-Klein塔
    ax3.text(0.1, 0.25, 'KK Tower:', fontsize=11, fontweight='bold')
    
    kk_levels = [1, 2, 3, 5, 8]  # 满足no-11的KK模式
    for i, n in enumerate(kk_levels):
        y = 0.18 - i * 0.03
        ax3.hlines(0.15, 0.2, 0.35, colors=COLORS['compact'], 
                  linewidth=2, alpha=0.8 - i*0.1)
        ax3.text(0.37, y, f'n={n}: m = {n}/R', ha='left', va='center',
                fontsize=9)
    
    # 景观约化
    landscape_box = FancyBboxPatch((0.6, 0.05), 0.35, 0.2,
                                  boxstyle="round,pad=0.02",
                                  facecolor=COLORS['landscape'],
                                  edgecolor='black',
                                  alpha=0.6)
    ax3.add_patch(landscape_box)
    ax3.text(0.775, 0.18, 'Landscape Reduction', ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')
    ax3.text(0.775, 0.13, 'Standard: ~10⁵⁰⁰ vacua', ha='center', va='center',
            fontsize=9, color='white')
    ax3.text(0.775, 0.08, 'With no-11: << 10³ vacua', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 0.85)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('dbrane_compactification_T14_3.png', dpi=300, bbox_inches='tight',
                facecolor=COLORS['background'])
    plt.close()

def main():
    """生成所有图表"""
    print("生成T14-3 φ-超对称与弦理论可视化...")
    
    # 生成图表
    create_supersymmetry_diagram()
    print("✓ 超对称图已生成: supersymmetry_T14_3.png")
    
    create_string_modes_diagram()
    print("✓ 弦振动模式图已生成: string_modes_T14_3.png")
    
    create_dbrane_compactification_diagram()
    print("✓ D-膜与紧致化图已生成: dbrane_compactification_T14_3.png")
    
    print("\n所有T14-3可视化图表已生成完成！")

if __name__ == '__main__':
    main()