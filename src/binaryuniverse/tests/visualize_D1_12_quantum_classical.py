#!/usr/bin/env python3
"""
D1.12 量子-经典边界可视化
展示Zeckendorf编码下的量子态坍缩和熵增
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fibonacci(n):
    """计算第n个Fibonacci数"""
    if n <= 0:
        return 0
    if n == 1 or n == 2:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def visualize_quantum_classical_boundary():
    """可视化量子-经典边界的核心概念"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 量子态的Zeckendorf编码
    ax1 = plt.subplot(2, 3, 1)
    
    # 量子叠加态的Fibonacci表示
    fib_indices = [1, 3, 5, 8]  # 非连续Fibonacci索引
    fib_values = [fibonacci(i) for i in fib_indices]
    
    ax1.bar(range(len(fib_indices)), fib_values, color='blue', alpha=0.7)
    ax1.set_xticks(range(len(fib_indices)))
    ax1.set_xticklabels([f'F_{i}' for i in fib_indices])
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Quantum State Zeckendorf Encoding\n|ψ⟩ = Σ αi Fi (No-11 satisfied)')
    ax1.grid(True, alpha=0.3)
    
    # 2. No-11约束违反与修复
    ax2 = plt.subplot(2, 3, 2)
    
    # 违反No-11的情况
    violated = [2, 3, 5, 6, 8]  # 有连续Fibonacci
    repaired = [4, 7, 8]  # 修复后
    
    x_pos = np.arange(10)
    violated_bar = np.zeros(10)
    repaired_bar = np.zeros(10)
    
    for v in violated:
        violated_bar[v-1] = 1
    for r in repaired:
        repaired_bar[r-1] = 1
    
    width = 0.35
    ax2.bar(x_pos - width/2, violated_bar, width, label='Violated (11 pattern)', color='red', alpha=0.6)
    ax2.bar(x_pos + width/2, repaired_bar, width, label='Repaired', color='green', alpha=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'F{i+1}' for i in range(10)], fontsize=8)
    ax2.set_ylabel('Presence')
    ax2.set_title('No-11 Constraint Repair\nFi + Fi+1 → Fi+2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 测量坍缩过程
    ax3 = plt.subplot(2, 3, 3)
    
    # 坍缩前后的概率分布
    states = ['|0⟩', '|1⟩', '|2⟩', '|3⟩']
    before_probs = [0.25, 0.35, 0.25, 0.15]  # 叠加态
    after_probs = [0, 1, 0, 0]  # 坍缩到|1⟩
    
    x = np.arange(len(states))
    width = 0.35
    
    ax3.bar(x - width/2, before_probs, width, label='Before measurement', color='purple', alpha=0.6)
    ax3.bar(x + width/2, after_probs, width, label='After collapse', color='orange', alpha=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(states)
    ax3.set_ylabel('Probability')
    ax3.set_title('Measurement Collapse\n|ψ⟩ → |mk⟩')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 熵增过程
    ax4 = plt.subplot(2, 3, 4)
    
    # 时间演化的熵
    t = np.linspace(0, 5, 100)
    phi = (1 + np.sqrt(5)) / 2
    
    # 量子系统的熵演化
    S_quantum = np.log(1 + t) / np.log(phi)
    # 经典系统的熵演化
    S_classical = t / phi
    # 测量导致的熵跳跃
    S_measurement = S_quantum.copy()
    jump_points = [20, 50, 80]
    for jp in jump_points:
        S_measurement[jp:] += 0.5
    
    ax4.plot(t, S_quantum, 'b-', label='Quantum evolution', linewidth=2)
    ax4.plot(t, S_classical, 'r--', label='Classical evolution', linewidth=2)
    ax4.plot(t, S_measurement, 'g:', label='With measurements', linewidth=2)
    ax4.set_xlabel('Time (φ units)')
    ax4.set_ylabel('Entropy Sφ')
    ax4.set_title('Entropy Increase: ΔSφ ≥ 1')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 量子-经典复杂度边界
    ax5 = plt.subplot(2, 3, 5)
    
    # 复杂度相空间
    q_complexity = np.linspace(0, 3, 50)
    c_threshold = phi * np.ones_like(q_complexity)
    
    # 量子和经典区域
    ax5.fill_between(q_complexity, 0, c_threshold, where=(q_complexity < phi),
                     color='blue', alpha=0.3, label='Classical regime')
    ax5.fill_between(q_complexity, c_threshold, 3, where=(q_complexity >= phi),
                     color='red', alpha=0.3, label='Quantum regime')
    
    ax5.plot(q_complexity, c_threshold, 'k--', linewidth=2, label=f'Boundary: Qφ = φ')
    ax5.axvline(x=phi, color='gold', linestyle=':', linewidth=2, label=f'φ = {phi:.3f}')
    
    ax5.set_xlabel('Quantum Complexity Qφ')
    ax5.set_ylabel('Classical Threshold')
    ax5.set_title('Quantum-Classical Boundary\nQφ < φ·Cφ → Classical')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 纠缠态的Fibonacci结构
    ax6 = plt.subplot(2, 3, 6)
    
    # Bell态的Zeckendorf表示
    bell_indices_A = [1, 3]  # 子系统A
    bell_indices_B = [2, 5]  # 子系统B
    
    # 创建纠缠图
    theta = np.linspace(0, 2*np.pi, 100)
    r1, r2 = 1, 0.6
    
    # 外圈 - 系统A
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    ax6.plot(x1, y1, 'b-', linewidth=2, label='System A')
    
    # 内圈 - 系统B
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    ax6.plot(x2, y2, 'r-', linewidth=2, label='System B')
    
    # 纠缠连线
    for i in range(0, 360, 45):
        angle = np.radians(i)
        ax6.plot([r2*np.cos(angle), r1*np.cos(angle)],
                [r2*np.sin(angle), r1*np.sin(angle)],
                'g--', alpha=0.5, linewidth=1)
    
    # 标记Fibonacci索引
    for i, idx in enumerate(bell_indices_A):
        angle = 2*np.pi * i / len(bell_indices_A)
        ax6.plot(r1*np.cos(angle), r1*np.sin(angle), 'bo', markersize=10)
        ax6.text(1.2*r1*np.cos(angle), 1.2*r1*np.sin(angle), f'F{idx}', fontsize=10)
    
    for i, idx in enumerate(bell_indices_B):
        angle = 2*np.pi * i / len(bell_indices_B) + np.pi/4
        ax6.plot(r2*np.cos(angle), r2*np.sin(angle), 'ro', markersize=10)
        ax6.text(0.4*r2*np.cos(angle), 0.4*r2*np.sin(angle), f'F{idx}', fontsize=10)
    
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_aspect('equal')
    ax6.set_title('Entangled State Structure\n|Φ+⟩ = (F1⊗F2 + F3⊗F5)/√2')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('D1.12: Quantum-Classical Boundary in Zeckendorf Encoding', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('D1_12_quantum_classical_boundary.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as D1_12_quantum_classical_boundary.png")
    
    plt.show()

def visualize_measurement_process():
    """详细可视化测量过程"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    phi = (1 + np.sqrt(5)) / 2
    
    # 1. 初始量子态
    ax = axes[0, 0]
    
    # Bloch球表示
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.1, color='gray')
    
    # 量子态矢量
    theta, phi_angle = np.pi/4, np.pi/6
    state_x = np.sin(theta) * np.cos(phi_angle)
    state_y = np.sin(theta) * np.sin(phi_angle)
    state_z = np.cos(theta)
    
    ax.quiver(0, 0, 0, state_x, state_y, state_z, color='blue', arrow_length_ratio=0.1, linewidth=3)
    ax.text(state_x, state_y, state_z, '|ψ⟩', fontsize=12)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Quantum State on Bloch Sphere')
    
    # 2. No-11约束检测
    ax = axes[0, 1]
    
    # 二进制表示
    binary_states = ['|0⟩: 1010001', '|1⟩: 0100100', 'Superposition: 1110101']
    colors = ['green', 'green', 'red']
    labels = ['Valid (No 11)', 'Valid (No 11)', 'Invalid (Contains 11)']
    
    for i, (state, color, label) in enumerate(zip(binary_states, colors, labels)):
        y_pos = 2 - i * 0.5
        ax.text(0.1, y_pos, state, fontsize=10)
        
        # 高亮11模式
        if '11' in state:
            idx = state.index('11')
            rect = Rectangle((0.5 + idx*0.08, y_pos-0.1), 0.16, 0.2, 
                           facecolor='red', alpha=0.3)
            ax.add_patch(rect)
        
        ax.text(1.2, y_pos, label, fontsize=10, color=color)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    ax.set_title('No-11 Constraint Detection')
    
    # 3. 坍缩动力学
    ax = axes[1, 0]
    
    t = np.linspace(0, 1, 100)
    
    # 概率演化
    p0 = 0.5 * (1 + np.exp(-10*(t-0.5)))  # Sigmoid坍缩
    p1 = 1 - p0
    
    ax.plot(t, p0, 'b-', linewidth=2, label='P(|0⟩)')
    ax.plot(t, p1, 'r-', linewidth=2, label='P(|1⟩)')
    ax.axvline(x=0.5, color='gold', linestyle='--', label='Measurement')
    
    ax.fill_between(t[:50], 0, 1, alpha=0.1, color='blue', label='Quantum')
    ax.fill_between(t[50:], 0, 1, alpha=0.1, color='red', label='Classical')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title('Collapse Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 熵的详细演化
    ax = axes[1, 1]
    
    # 不同阶段的熵
    stages = ['Pure\nState', 'Measurement\nInteraction', 'No-11\nRepair', 'Classical\nState']
    entropy_values = [0, 0.5, 1.2, 0.3]
    colors_bar = ['blue', 'purple', 'orange', 'red']
    
    bars = ax.bar(stages, entropy_values, color=colors_bar, alpha=0.7)
    
    # 添加熵增箭头
    for i in range(len(stages)-1):
        ax.annotate('', xy=(i+1, entropy_values[i+1]), xytext=(i+0.2, entropy_values[i]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_ylabel('Entropy Sφ')
    ax.set_title('Entropy Evolution During Measurement')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加总熵增标注
    ax.text(1.5, 1.5, f'Total ΔSφ ≥ 1', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Measurement Process and Entropy Increase', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('D1_12_measurement_process.png', dpi=300, bbox_inches='tight')
    print("Measurement process saved as D1_12_measurement_process.png")
    
    plt.show()

if __name__ == '__main__':
    print("Generating D1.12 Quantum-Classical Boundary visualizations...")
    
    # 主要概念可视化
    visualize_quantum_classical_boundary()
    
    # 测量过程详细可视化
    # visualize_measurement_process()
    
    print("All visualizations completed!")