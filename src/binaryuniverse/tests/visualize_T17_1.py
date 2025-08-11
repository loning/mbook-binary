#!/usr/bin/env python3
"""
T17-1 φ-弦对偶性定理可视化
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

# 黄金比例
phi = 1.618033988749895

# ==================== 图1: T对偶变换图示 ====================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('T-Duality Transformation: R ↔ α\'/R', fontsize=16, fontweight='bold')

# 绘制T对偶变换
R_values = np.array([phi**i for i in range(-3, 4)])
R_dual_values = 1.0 / R_values

# 绘制半径值
x_pos = np.arange(len(R_values))
bar_width = 0.35

bars1 = ax1.bar(x_pos - bar_width/2, R_values, bar_width, label='Original R', alpha=0.7, color='blue')
bars2 = ax1.bar(x_pos + bar_width/2, R_dual_values, bar_width, label='Dual R\'', alpha=0.7, color='red')

# 添加对偶箭头
for i in range(len(x_pos)):
    ax1.annotate('', xy=(x_pos[i] + bar_width/2, R_dual_values[i] + 0.1), 
                xytext=(x_pos[i] - bar_width/2, R_values[i] + 0.1),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))

ax1.set_xlabel('φ-power index', fontsize=12)
ax1.set_ylabel('Radius value', fontsize=12)
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ==================== 图2: S对偶变换图示 ====================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('S-Duality: g_s ↔ 1/g_s', fontsize=16, fontweight='bold')

# 绘制耦合常数变换
g_values = np.linspace(0.1, 3, 50)
g_dual = 1.0 / g_values

ax2.plot(g_values, g_dual, 'b-', linewidth=3, label='g\' = 1/g')
ax2.plot(g_values, g_values, 'k--', linewidth=1, alpha=0.5, label='g\' = g (自对偶线)')
ax2.axvline(x=1, color='red', linestyle=':', linewidth=2, label='自对偶点 g=1')
ax2.axhline(y=1, color='red', linestyle=':', linewidth=1, alpha=0.5)

# 标记弱耦合和强耦合区域
ax2.fill_between([0.1, 1], [0, 0], [10, 10], alpha=0.2, color='green', label='弱耦合区')
ax2.fill_between([1, 3], [0, 0], [10, 10], alpha=0.2, color='orange', label='强耦合区')

ax2.set_xlabel('g_s', fontsize=12)
ax2.set_ylabel('g_s\'', fontsize=12)
ax2.set_xlim(0.1, 3)
ax2.set_ylim(0.1, 10)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# ==================== 图3: 对偶群结构 ====================
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Duality Group Structure Γ^φ ⊂ SL(2,Z)', fontsize=16, fontweight='bold')

# 绘制模空间基本域
theta = np.linspace(0, 2*np.pi, 100)
r = 1
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta) + 1

# 基本域边界
ax3.plot([-0.5, -0.5], [0, 2], 'b-', linewidth=2)
ax3.plot([0.5, 0.5], [0, 2], 'b-', linewidth=2)
ax3.plot(x_circle[x_circle >= -0.5], y_circle[x_circle >= -0.5], 'b-', linewidth=2)
ax3.plot(x_circle[x_circle <= 0.5], y_circle[x_circle <= 0.5], 'b-', linewidth=2)

# 填充基本域
ax3.fill_between([-0.5, 0.5], [0, 0], [2, 2], alpha=0.3, color='lightblue')

# 标记特殊点
ax3.plot(0, 1, 'ro', markersize=10, label='τ = i')
ax3.plot(-0.5, np.sqrt(3)/2, 'go', markersize=10, label='τ = e^{2πi/3}')
ax3.plot(0.5, np.sqrt(3)/2, 'go', markersize=10)

# 添加变换箭头
ax3.annotate('T: τ → τ+1', xy=(0.5, 1.5), xytext=(-0.5, 1.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.annotate('S: τ → -1/τ', xy=(0, 0.5), xytext=(0, 1.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

ax3.set_xlabel('Re(τ)', fontsize=12)
ax3.set_ylabel('Im(τ)', fontsize=12)
ax3.set_xlim(-1, 1)
ax3.set_ylim(0, 2.5)
ax3.legend()
ax3.grid(True, alpha=0.3)

# ==================== 图4: 对偶链与熵增 ====================
ax4 = fig.add_subplot(gs[2, :])
ax4.set_title('Duality Chain and Entropy Increase', fontsize=16, fontweight='bold')

# 创建对偶链
chain_length = 6
x_positions = np.arange(chain_length)
entropy_values = [3.0 + 0.5*i + 0.1*i**2 for i in range(chain_length)]

# 绘制配置节点
for i, (x, s) in enumerate(zip(x_positions, entropy_values)):
    circle = Circle((x, 0), 0.3, color='lightblue', ec='blue', linewidth=2)
    ax4.add_patch(circle)
    ax4.text(x, 0, f'S{i}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 熵值标签
    ax4.text(x, -0.8, f'S={s:.2f}', ha='center', va='center', fontsize=9)
    
    # 对偶变换箭头
    if i < chain_length - 1:
        arrow = FancyArrowPatch((x + 0.3, 0), (x + 0.7, 0),
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', 
                               mutation_scale=20, 
                               linewidth=2,
                               color='green' if i % 2 == 0 else 'red')
        ax4.add_patch(arrow)
        
        # 变换标签
        transform_label = 'T' if i % 2 == 0 else 'S'
        ax4.text(x + 0.5, 0.4, transform_label, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='green' if i % 2 == 0 else 'red')

# 绘制熵增曲线
ax4_twin = ax4.twinx()
ax4_twin.plot(x_positions, entropy_values, 'k-', linewidth=2, marker='o', markersize=8)
ax4_twin.fill_between(x_positions, 0, entropy_values, alpha=0.2, color='orange')
ax4_twin.set_ylabel('Entropy S[Configuration]', fontsize=12)
ax4_twin.set_ylim(0, max(entropy_values) * 1.2)

ax4.set_xlim(-0.5, chain_length - 0.5)
ax4.set_ylim(-1.5, 1)
ax4.set_xlabel('Duality Transformation Steps', fontsize=12)
ax4.set_xticks(x_positions)
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_yticks([])

# 添加熵增箭头
ax4_twin.annotate('Entropy Increase', xy=(chain_length-1, entropy_values[-1]), 
                  xytext=(chain_length-2, entropy_values[-1]*0.8),
                  arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                  fontsize=12, fontweight='bold')

# ==================== 图5: Mirror对称性 ====================
ax5 = fig.add_subplot(gs[3, 0])
ax5.set_title('Mirror Symmetry: Hodge Diamond Exchange', fontsize=16, fontweight='bold')

# 绘制两个Hodge钻石
def draw_hodge_diamond(ax, x_center, y_center, h11, h21, label):
    # 钻石形状的坐标
    diamond_x = [x_center, x_center+0.3, x_center, x_center-0.3, x_center]
    diamond_y = [y_center+0.5, y_center, y_center-0.5, y_center, y_center+0.5]
    
    # 绘制钻石
    ax.plot(diamond_x, diamond_y, 'b-', linewidth=2)
    ax.fill(diamond_x, diamond_y, alpha=0.3, color='lightblue')
    
    # 添加Hodge数
    ax.text(x_center, y_center+0.3, '1', ha='center', va='center', fontsize=10)
    ax.text(x_center-0.2, y_center, f'{h11}', ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    ax.text(x_center+0.2, y_center, f'{h21}', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    ax.text(x_center, y_center-0.3, '1', ha='center', va='center', fontsize=10)
    
    # 标签
    ax.text(x_center, y_center-0.8, label, ha='center', va='center', fontsize=12, fontweight='bold')

# CY1和CY2
draw_hodge_diamond(ax5, 0, 0, 3, 243, 'CY₁')
draw_hodge_diamond(ax5, 2, 0, 243, 3, 'CY₂')

# Mirror箭头
arrow = FancyArrowPatch((0.4, 0), (1.6, 0),
                       connectionstyle="arc3,rad=0.3", 
                       arrowstyle='<->', 
                       mutation_scale=25, 
                       linewidth=3,
                       color='green')
ax5.add_patch(arrow)
ax5.text(1, 0.3, 'Mirror', ha='center', va='center', fontsize=12, fontweight='bold', color='green')

# 添加说明
ax5.text(1, -1.5, 'h¹¹ ↔ h²¹', ha='center', va='center', fontsize=14, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

ax5.set_xlim(-0.8, 2.8)
ax5.set_ylim(-2, 1)
ax5.axis('off')

# ==================== 图6: BPS谱与对偶不变量 ====================
ax6 = fig.add_subplot(gs[3, 1])
ax6.set_title('BPS Spectrum and Duality Invariants', fontsize=16, fontweight='bold')

# BPS态质量谱
masses = [1.0, phi, phi**2, phi**3]
charges = [(1, 0), (1, 1), (2, 1), (3, 2)]
degeneracies = [1, 2, 3, 5]

# 绘制BPS谱
for i, (m, (p, q), d) in enumerate(zip(masses, charges, degeneracies)):
    # 质量线
    ax6.hlines(m, i-0.3, i+0.3, colors='blue', linewidth=3)
    
    # 简并度
    for j in range(d):
        x = i - 0.2 + 0.1*j
        ax6.plot(x, m, 'ro', markersize=6)
    
    # 电荷标签
    ax6.text(i, m+0.1, f'({p},{q})', ha='center', va='bottom', fontsize=9)

# 对偶不变量
ax6.axhline(y=15, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax6.text(3.5, 15, 'Central Charge c=15', ha='right', va='bottom', fontsize=10, color='green')

ax6.set_xlabel('BPS State Index', fontsize=12)
ax6.set_ylabel('Mass / Central Charge', fontsize=12)
ax6.set_xlim(-0.5, 3.5)
ax6.set_ylim(0, 20)
ax6.grid(True, alpha=0.3)

# 添加图例
red_dot = mpatches.Patch(color='red', label='BPS states')
blue_line = mpatches.Patch(color='blue', label='Mass level')
green_line = mpatches.Patch(color='green', label='Invariant')
ax6.legend(handles=[red_dot, blue_line, green_line], loc='upper left')

# ==================== 总标题 ====================
fig.suptitle('T17-1: φ-String Duality Theory Visualization', fontsize=20, fontweight='bold', y=0.98)

# 保存图片
plt.tight_layout()
plt.savefig('T17_1_string_duality_visualization.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("T17-1可视化图表已生成: T17_1_string_duality_visualization.png")
plt.close()

# ==================== 额外图: 对偶网络结构 ====================
fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_title('Duality Network Structure in φ-Encoded Universe', fontsize=16, fontweight='bold')

# 生成网络节点
np.random.seed(42)
n_nodes = 20
angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
radii = 3 + 0.5*np.random.randn(n_nodes)
x_nodes = radii * np.cos(angles)
y_nodes = radii * np.sin(angles)

# 绘制节点
for i, (x, y) in enumerate(zip(x_nodes, y_nodes)):
    color = 'lightblue' if i % 3 == 0 else 'lightgreen' if i % 3 == 1 else 'lightyellow'
    circle = Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, f'S{i}', ha='center', va='center', fontsize=8)

# 绘制连接（对偶变换）
connections = [(0, 3), (0, 7), (1, 4), (2, 5), (3, 6), (4, 8), (5, 9), (6, 10),
               (7, 11), (8, 12), (9, 13), (10, 14), (11, 15), (12, 16), (13, 17),
               (14, 18), (15, 19), (16, 0), (17, 1), (18, 2), (19, 3)]

for i, j in connections:
    if i < len(x_nodes) and j < len(x_nodes):
        # T对偶用绿色，S对偶用红色
        color = 'green' if (i + j) % 2 == 0 else 'red'
        style = '-' if (i + j) % 2 == 0 else '--'
        ax.plot([x_nodes[i], x_nodes[j]], [y_nodes[i], y_nodes[j]], 
                color=color, linestyle=style, linewidth=1.5, alpha=0.6)

# 添加禁区
forbidden_angles = [np.pi/6, 5*np.pi/6, 3*np.pi/2]
for angle in forbidden_angles:
    x = 5 * np.cos(angle)
    y = 5 * np.sin(angle)
    forbidden = Circle((x, y), 0.8, color='red', alpha=0.3)
    ax.add_patch(forbidden)
    ax.text(x, y, 'Forbidden\nby no-11', ha='center', va='center', fontsize=9, color='darkred')

# 图例
t_line = mpatches.Patch(color='green', label='T-duality')
s_line = mpatches.Patch(color='red', label='S-duality')
forbidden_patch = mpatches.Patch(color='red', alpha=0.3, label='Forbidden region')
ax.legend(handles=[t_line, s_line, forbidden_patch], loc='upper right')

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('Configuration Space X', fontsize=12)
ax.set_ylabel('Configuration Space Y', fontsize=12)

plt.tight_layout()
plt.savefig('T17_1_duality_network.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("对偶网络结构图已生成: T17_1_duality_network.png")
plt.close()

print("\n所有T17-1可视化图表生成完成！")