#!/usr/bin/env python3
"""
T14-1 φ-规范场理论可视化程序
展示φ-编码规范场的关键特性
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import cm
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_phi_gauge_symmetry():
    """可视化φ-规范对称性与递归自指结构"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('T14-1: φ-规范场理论的递归自指结构', fontsize=16)
    
    # 1. 规范变换的φ-编码表示
    ax1 = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 原始规范场配置（φ-编码）
    r_original = 1 + 0.2 * np.sin(5*theta)
    ax1.plot(theta, r_original, 'b-', linewidth=2, label='原始场 A^φ')
    
    # 规范变换后的配置
    gauge_param = 0.3
    r_transformed = r_original * (1 + gauge_param * np.cos(3*theta))
    ax1.plot(theta, r_transformed, 'r--', linewidth=2, label='变换后 A^φ + δA^φ')
    
    # 物理可观测量（规范不变）
    observable = np.mean(r_original) * np.ones_like(theta)
    ax1.plot(theta, observable, 'g:', linewidth=3, label='物理量（不变）')
    
    ax1.set_xlabel('相位 θ')
    ax1.set_ylabel('场强度')
    ax1.set_title('规范变换下的φ-编码场')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 递归自指结构的熵演化
    ax2 = axes[0, 1]
    n_steps = 50
    entropy = np.zeros(n_steps)
    recursive_depth = np.zeros(n_steps)
    
    # 模拟递归自指演化
    psi = 1.0  # 初始状态
    for i in range(n_steps):
        # ψ = ψ(ψ) 递归
        psi = psi * (1 + 0.1 * np.sin(psi))
        recursive_depth[i] = np.log(abs(psi) + 1)
        # 熵增公理：自指完备系统必然熵增
        entropy[i] = entropy[i-1] + 0.1 * recursive_depth[i] if i > 0 else 0
    
    ax2.plot(range(n_steps), entropy, 'b-', linewidth=2, label='规范对称性熵 S^φ')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(n_steps), recursive_depth, 'r--', linewidth=2, label='递归深度')
    
    ax2.set_xlabel('演化时间 τ')
    ax2.set_ylabel('熵 S^φ', color='b')
    ax2_twin.set_ylabel('递归深度', color='r')
    ax2.set_title('熵增与递归自指')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # 添加图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 3. Yang-Mills场强张量的φ-编码
    ax3 = axes[1, 0]
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    
    # 场强张量 F^{μν,φ} 的模拟（满足no-11约束）
    F_field = np.sin(X) * np.cos(Y) * np.exp(-0.2*(X**2 + Y**2))
    
    # 确保no-11约束：避免连续的高场强区域
    for i in range(1, F_field.shape[0]-1):
        for j in range(1, F_field.shape[1]-1):
            if abs(F_field[i,j]) > 0.8 and abs(F_field[i+1,j]) > 0.8:
                F_field[i+1,j] *= 0.5  # 降低以避免"11"模式
    
    im = ax3.contourf(X, Y, F_field, levels=20, cmap='RdBu_r')
    ax3.contour(X, Y, F_field, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # 添加矢量场箭头
    skip = 5
    U = -np.gradient(F_field, axis=1)[::skip, ::skip]
    V = np.gradient(F_field, axis=0)[::skip, ::skip]
    ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip], U, V, 
              alpha=0.5, width=0.003)
    
    ax3.set_xlabel('空间坐标 x')
    ax3.set_ylabel('空间坐标 y')
    ax3.set_title('φ-Yang-Mills场强张量 F^{μν,φ}')
    plt.colorbar(im, ax=ax3, label='场强')
    
    # 4. BRST对称性与ghost场
    ax4 = axes[1, 1]
    
    # 创建BRST复形图
    levels = ['物理态', 'ghost态', 'antighost态', 'auxiliary态']
    y_pos = [3, 2, 1, 0]
    
    # 绘制态
    for i, (level, y) in enumerate(zip(levels, y_pos)):
        circle = Circle((0.5, y), 0.3, color=plt.cm.viridis(i/3), alpha=0.6)
        ax4.add_patch(circle)
        ax4.text(0.5, y, level, ha='center', va='center', fontsize=10, 
                weight='bold')
    
    # 绘制BRST变换箭头
    arrows = [
        FancyArrowPatch((0.5, 2.7), (0.5, 2.3), 
                       connectionstyle="arc3,rad=.3", 
                       arrowstyle='->', mutation_scale=20, lw=2),
        FancyArrowPatch((0.5, 1.7), (0.5, 1.3), 
                       connectionstyle="arc3,rad=-.3", 
                       arrowstyle='->', mutation_scale=20, lw=2),
        FancyArrowPatch((0.5, 0.7), (0.5, 0.3), 
                       connectionstyle="arc3,rad=.3", 
                       arrowstyle='->', mutation_scale=20, lw=2)
    ]
    
    for arrow in arrows:
        ax4.add_patch(arrow)
    
    # 添加BRST幂零性标注
    ax4.text(1.0, 1.5, 'Q²=0', fontsize=14, weight='bold', color='red')
    ax4.text(0.0, 1.5, 's', fontsize=12, style='italic')
    
    ax4.set_xlim(-0.5, 1.5)
    ax4.set_ylim(-0.5, 3.5)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('BRST对称性与量子修正')
    
    plt.tight_layout()
    return fig

def visualize_no11_constraint_in_gauge():
    """可视化no-11约束在规范场中的作用"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('no-11约束在φ-规范场理论中的物理意义', fontsize=16)
    
    # 1. 允许的场配置
    ax1 = axes[0]
    field_config = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    x = range(len(field_config))
    colors = ['green' if field_config[i] == 1 else 'lightgray' for i in x]
    
    ax1.bar(x, field_config, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylim(-0.1, 1.5)
    ax1.set_xlabel('Fibonacci索引')
    ax1.set_ylabel('场分量激发')
    ax1.set_title('允许的φ-规范场配置（无11）')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 标注Fibonacci数
    fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    for i, fib in enumerate(fib_numbers[:len(field_config)]):
        ax1.text(i, -0.05, f'F_{i}\n={fib}', ha='center', va='top', fontsize=8)
    
    # 2. 因果结构保持
    ax2 = axes[1]
    t = np.linspace(0, 10, 1000)
    
    # 正常传播（满足no-11）
    signal_good = np.exp(-0.5*t) * np.sin(2*np.pi*t)
    ax2.plot(t, signal_good, 'b-', linewidth=2, label='因果传播（no-11）')
    
    # 非因果传播（违反no-11，会被抑制）
    signal_bad = np.exp(-0.5*t) * np.sin(2*np.pi*t) * (1 + 0.5*np.sin(20*np.pi*t))
    # 在违反点衰减
    for i in range(1, len(t)-1):
        if abs(signal_bad[i]) > 0.5 and abs(signal_bad[i+1]) > 0.5:
            signal_bad[i+1:] *= 0.3
            break
    
    ax2.plot(t, signal_bad, 'r--', linewidth=2, label='非因果（被抑制）')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    
    ax2.set_xlabel('时间 t')
    ax2.set_ylabel('场振幅')
    ax2.set_title('no-11约束确保因果传播')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 规范不变性等价性
    ax3 = axes[2]
    
    # 创建关系图
    concepts = ['no-11约束', '因果性', '规范不变', '递归稳定']
    positions = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    # 绘制概念节点
    for concept, pos in zip(concepts, positions):
        circle = Circle(pos, 0.2, color='lightblue', edgecolor='darkblue', 
                       linewidth=2, alpha=0.8)
        ax3.add_patch(circle)
        ax3.text(pos[0], pos[1], concept, ha='center', va='center', 
                fontsize=10, weight='bold')
    
    # 绘制等价关系
    connections = [
        (positions[0], positions[1]),
        (positions[1], positions[2]),
        (positions[2], positions[3]),
        (positions[3], positions[0]),
        (positions[0], positions[2]),
        (positions[1], positions[3])
    ]
    
    for start, end in connections:
        ax3.plot([start[0], end[0]], [start[1], end[1]], 
                'k-', linewidth=1, alpha=0.5)
    
    # 中心标注
    ax3.text(0.5, 0.5, '⟺', fontsize=20, ha='center', va='center', 
            weight='bold', color='red')
    
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('定理：约束⟺对称性⟺稳定性')
    
    plt.tight_layout()
    return fig

def visualize_gauge_entropy_evolution():
    """可视化规范场的熵演化"""
    fig = plt.figure(figsize=(12, 8))
    
    # 创建3D子图
    ax = fig.add_subplot(111, projection='3d')
    
    # 参数设置
    n_steps = 100
    n_fields = 20
    
    # 初始化场配置
    fields = np.random.rand(n_fields, n_steps) * 0.1
    entropy = np.zeros((n_fields, n_steps))
    
    # 模拟演化
    for t in range(1, n_steps):
        for i in range(n_fields):
            # 规范场演化（简化模型）
            neighbors = []
            if i > 0: neighbors.append(fields[i-1, t-1])
            if i < n_fields-1: neighbors.append(fields[i+1, t-1])
            
            # 自指演化：场依赖于自身和邻居
            fields[i, t] = fields[i, t-1] * (1 + 0.1 * np.mean(neighbors))
            
            # 熵计算（累积）
            entropy[i, t] = entropy[i, t-1] + 0.01 * abs(fields[i, t])
            
            # 应用no-11约束
            if i > 0 and abs(fields[i, t]) > 0.8 and abs(fields[i-1, t]) > 0.8:
                fields[i, t] *= 0.5
    
    # 创建网格
    T, I = np.meshgrid(range(n_steps), range(n_fields))
    
    # 绘制3D表面
    surf = ax.plot_surface(T, I, entropy, cmap='viridis', 
                          edgecolor='none', alpha=0.8)
    
    # 添加等高线投影
    contours = ax.contour(T, I, entropy, zdir='z', offset=0, 
                         levels=10, cmap='viridis', alpha=0.5)
    
    ax.set_xlabel('演化时间 τ')
    ax.set_ylabel('场分量索引')
    ax.set_zlabel('规范对称性熵 S^φ')
    ax.set_title('φ-规范场的熵增演化\n（唯一公理：自指完备系统必然熵增）')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    return fig

# 生成所有可视化
if __name__ == "__main__":
    # 1. 规范对称性与递归自指
    fig1 = visualize_phi_gauge_symmetry()
    fig1.savefig('phi_gauge_symmetry_T14_1.png', dpi=300, bbox_inches='tight')
    print("已生成：phi_gauge_symmetry_T14_1.png")
    
    # 2. no-11约束的物理意义
    fig2 = visualize_no11_constraint_in_gauge()
    fig2.savefig('no11_constraint_gauge_T14_1.png', dpi=300, bbox_inches='tight')
    print("已生成：no11_constraint_gauge_T14_1.png")
    
    # 3. 规范场的熵演化
    fig3 = visualize_gauge_entropy_evolution()
    fig3.savefig('gauge_entropy_evolution_T14_1.png', dpi=300, bbox_inches='tight')
    print("已生成：gauge_entropy_evolution_T14_1.png")
    
    plt.show()