#!/usr/bin/env python3
"""
T0-11递归深度与层级理论的3D可视化系统
展示从Zeckendorf编码中涌现的层级结构和φ-缩放动画

核心功能:
1. Fibonacci层级的3D结构可视化
2. φ-缩放动画和复杂性增长曲面
3. 递归深度的量子化演示
4. 信息流收敛的3D动画
5. 不可逆性转换的层级跳跃
6. 相变点的临界现象可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import math
import colorsys
from matplotlib.colors import ListedColormap

# 设置全局样式
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5


class RecursiveHierarchySystem:
    """递归层级系统"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # 黄金比例
        self.tau_0 = 1.0  # 时间量子
        
        # 预计算Fibonacci数列
        self.fibonacci = self._generate_fibonacci(50)
        
        # 系统状态
        self.states = []
        self.hierarchy_levels = {}
        self.complexity_surface = None
        
        # 相变参数
        self.critical_depths = []
        self.phase_transitions = []
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci序列"""
        fib = [1, 2]  # F_1=1, F_2=2避免退化
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def calculate_recursive_depth(self, initial_state: str = "1", target_state: str = "21") -> List[Dict]:
        """计算递归深度演化"""
        states = []
        current = 1
        depth = 0
        
        # 模拟递归深度增长
        while depth < 30:
            # 计算当前状态的属性
            zeckendorf = self._to_zeckendorf(current)
            level = self._determine_level(depth)
            entropy = self._calculate_entropy(depth)
            complexity = self._calculate_complexity(depth)
            
            # 检查No-11约束
            valid = self._check_no_11(zeckendorf)
            
            state = {
                'depth': depth,
                'value': current,
                'zeckendorf': zeckendorf,
                'level': level,
                'entropy': entropy,
                'complexity': complexity,
                'valid': valid,
                'phi_scaling': self.phi ** depth
            }
            
            states.append(state)
            
            # 递归增长
            current = self._apply_recursive_step(current)
            depth += 1
            
        return states
    
    def _to_zeckendorf(self, n: int) -> str:
        """转换为Zeckendorf表示"""
        if n <= 0:
            return "0"
        
        result = []
        i = len(self.fibonacci) - 1
        
        while i >= 0 and n > 0:
            if self.fibonacci[i] <= n:
                result.append('1')
                n -= self.fibonacci[i]
                i -= 2  # 避免连续1
            else:
                if result:
                    result.append('0')
                i -= 1
        
        return ''.join(result) if result else "0"
    
    def _check_no_11(self, zeck_str: str) -> bool:
        """检查No-11约束"""
        return "11" not in zeck_str
    
    def _determine_level(self, depth: int) -> int:
        """确定层级"""
        for k, F_k in enumerate(self.fibonacci):
            if depth < F_k:
                return k
        return len(self.fibonacci)
    
    def _calculate_entropy(self, depth: int) -> float:
        """计算深度熵"""
        if depth == 0:
            return 0.0
        return depth * math.log2(self.phi) - math.log2(math.sqrt(5))
    
    def _calculate_complexity(self, depth: int) -> float:
        """计算复杂性"""
        if depth < len(self.fibonacci):
            return float(self.fibonacci[depth])
        else:
            # 使用Binet公式近似
            return (self.phi ** (depth + 1)) / math.sqrt(5)
    
    def _apply_recursive_step(self, current: int) -> int:
        """应用递归步骤"""
        return current + 1  # 简化的递归增长
    
    def generate_hierarchy_levels(self, max_depth: int = 25) -> Dict[int, List[Dict]]:
        """生成层级结构"""
        states = self.calculate_recursive_depth()[:max_depth]
        levels = {}
        
        for state in states:
            level = state['level']
            if level not in levels:
                levels[level] = []
            levels[level].append(state)
        
        self.hierarchy_levels = levels
        return levels
    
    def create_complexity_surface(self, depth_range: Tuple[int, int], time_range: Tuple[int, int]) -> np.ndarray:
        """创建复杂性增长曲面"""
        depths = np.linspace(depth_range[0], depth_range[1], 50)
        times = np.linspace(time_range[0], time_range[1], 50)
        
        D, T = np.meshgrid(depths, times)
        
        # 复杂性函数 C(d,t) = φ^d * (1 + t/τ_0)
        C = np.power(self.phi, D) * (1 + T / self.tau_0)
        
        # 应用No-11修正
        for i, d in enumerate(depths):
            for j, t in enumerate(times):
                if not self._check_fibonacci_constraint(d, t):
                    C[j, i] *= 0.5  # 约束导致的抑制
        
        self.complexity_surface = C
        return C
    
    def _check_fibonacci_constraint(self, depth: float, time: float) -> bool:
        """检查Fibonacci约束"""
        # 简化的约束检查：避免非Fibonacci比例
        ratio = time / (depth + 1e-6)
        for i in range(len(self.fibonacci) - 1):
            fib_ratio = self.fibonacci[i+1] / self.fibonacci[i]
            if abs(ratio - fib_ratio) < 0.1:
                return True
        return False
    
    def find_phase_transitions(self) -> List[Dict]:
        """寻找相变点"""
        states = self.calculate_recursive_depth()
        transitions = []
        
        for i in range(1, len(states)):
            prev_level = states[i-1]['level']
            curr_level = states[i]['level']
            
            if curr_level != prev_level:
                # 发现层级跳跃
                entropy_jump = states[i]['entropy'] - states[i-1]['entropy']
                complexity_jump = states[i]['complexity'] - states[i-1]['complexity']
                
                transition = {
                    'depth': states[i]['depth'],
                    'from_level': prev_level,
                    'to_level': curr_level,
                    'entropy_jump': entropy_jump,
                    'complexity_jump': complexity_jump,
                    'phi_depth': self.phi ** states[i]['depth']
                }
                
                transitions.append(transition)
        
        self.phase_transitions = transitions
        return transitions


class Hierarchy3DVisualizer:
    """3D层级可视化器"""
    
    def __init__(self, system: RecursiveHierarchySystem):
        self.system = system
        self.fig = None
        self.axes = {}
        
        # 颜色映射
        self.level_colors = self._generate_level_colormap()
        
    def _generate_level_colormap(self) -> Dict[int, str]:
        """生成层级颜色映射"""
        colors = {}
        for i in range(10):  # 支持10个层级
            hue = (i * 0.618034) % 1  # 黄金角分布
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors[i] = rgb
        return colors
    
    def setup_3d_dashboard(self):
        """设置3D面板"""
        self.fig = plt.figure(figsize=(20, 16))
        self.fig.suptitle('T0-11: 递归深度与层级理论 - 3D可视化系统', fontsize=18, fontweight='bold')
        
        # 创建3D子图网格
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 主要3D层级结构
        self.axes['main_3d'] = self.fig.add_subplot(gs[0, :], projection='3d')
        self.axes['main_3d'].set_title('3D递归层级结构', fontsize=16)
        
        # 复杂性曲面
        self.axes['complexity_surface'] = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.axes['complexity_surface'].set_title('复杂性增长曲面', fontsize=14)
        
        # φ-缩放动画
        self.axes['phi_scaling'] = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.axes['phi_scaling'].set_title('φ-缩放动画', fontsize=14)
        
        # 相变点可视化
        self.axes['phase_transitions'] = self.fig.add_subplot(gs[1, 2])
        self.axes['phase_transitions'].set_title('相变点分析', fontsize=14)
        
        # 信息流网络
        self.axes['info_flow'] = self.fig.add_subplot(gs[2, 0], projection='3d')
        self.axes['info_flow'].set_title('信息流收敛', fontsize=14)
        
        # 不可逆性演示
        self.axes['irreversibility'] = self.fig.add_subplot(gs[2, 1])
        self.axes['irreversibility'].set_title('不可逆转换', fontsize=14)
        
        # 层级统计
        self.axes['level_stats'] = self.fig.add_subplot(gs[2, 2])
        self.axes['level_stats'].set_title('层级分布统计', fontsize=14)
    
    def visualize_3d_hierarchy(self, frame: int = 0):
        """3D层级结构可视化"""
        ax = self.axes['main_3d']
        ax.clear()
        ax.set_title('3D递归层级结构 - Fibonacci边界', fontsize=16)
        
        # 生成层级数据
        levels = self.system.generate_hierarchy_levels()
        
        # 绘制每个层级
        for level, states in levels.items():
            if not states:
                continue
            
            # 层级中心位置
            level_radius = level * 2 + 1
            level_height = level * 3
            
            # 在圆周上分布状态
            n_states = len(states)
            angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
            
            xs, ys, zs = [], [], []
            colors = []
            sizes = []
            
            for i, (angle, state) in enumerate(zip(angles, states)):
                x = level_radius * np.cos(angle)
                y = level_radius * np.sin(angle)
                z = level_height + state['entropy'] * 0.5
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
                
                # 颜色编码
                if state['valid']:
                    colors.append(self.level_colors.get(level, 'blue'))
                else:
                    colors.append('red')
                
                # 大小编码复杂性
                sizes.append(max(20, min(200, state['complexity'] * 2)))
            
            # 绘制状态点
            ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.7, 
                      label=f'Level {level}' if level < 5 else '')
            
            # 连接同层级状态
            if len(xs) > 1:
                for i in range(len(xs)):
                    j = (i + 1) % len(xs)
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 
                           'gray', alpha=0.3, linewidth=0.5)
            
            # 连接到下一层级
            if level + 1 in levels and levels[level + 1]:
                next_level_states = levels[level + 1]
                next_angles = np.linspace(0, 2*np.pi, len(next_level_states), endpoint=False)
                next_radius = (level + 1) * 2 + 1
                next_height = (level + 1) * 3
                
                for i, state in enumerate(states):
                    if i < len(next_level_states):  # 连接到下一层
                        next_x = next_radius * np.cos(next_angles[i])
                        next_y = next_radius * np.sin(next_angles[i])
                        next_z = next_height + next_level_states[i]['entropy'] * 0.5
                        
                        # 信息流箭头
                        ax.plot([xs[i], next_x], [ys[i], next_y], [zs[i], next_z], 
                               'blue', alpha=0.4, linewidth=1)
        
        # 绘制Fibonacci边界
        for k, F_k in enumerate(self.system.fibonacci[:6]):
            boundary_height = k * 3 + F_k * 0.1
            theta = np.linspace(0, 2*np.pi, 100)
            boundary_radius = k * 2 + 1
            
            x_boundary = boundary_radius * np.cos(theta)
            y_boundary = boundary_radius * np.sin(theta)
            z_boundary = np.full_like(theta, boundary_height)
            
            ax.plot(x_boundary, y_boundary, z_boundary, 'red', alpha=0.5, 
                   linewidth=2, linestyle='--')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('递归深度 + 熵')
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    def visualize_complexity_surface(self):
        """复杂性增长曲面"""
        ax = self.axes['complexity_surface']
        ax.clear()
        ax.set_title('复杂性增长曲面 C(d,t)', fontsize=14)
        
        # 创建复杂性曲面
        surface = self.system.create_complexity_surface((0, 15), (0, 10))
        
        depths = np.linspace(0, 15, 50)
        times = np.linspace(0, 10, 50)
        D, T = np.meshgrid(depths, times)
        
        # 绘制曲面
        surf = ax.plot_surface(D, T, np.log10(surface + 1), 
                              cmap='viridis', alpha=0.8)
        
        # 添加φ增长线
        phi_line_d = np.linspace(0, 15, 100)
        phi_line_c = np.log10(self.system.phi ** phi_line_d)
        phi_line_t = np.full_like(phi_line_d, 5)
        
        ax.plot(phi_line_d, phi_line_t, phi_line_c, 'red', linewidth=3, 
               label='φ^d增长线')
        
        ax.set_xlabel('递归深度 d')
        ax.set_ylabel('时间 t')
        ax.set_zlabel('log₁₀(复杂性)')
        
        # 颜色条
        self.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    def visualize_phi_scaling_animation(self, frame: int):
        """φ-缩放动画"""
        ax = self.axes['phi_scaling']
        ax.clear()
        ax.set_title(f'φ-缩放动画 (步骤 {frame})', fontsize=14)
        
        # 生成螺旋结构
        max_depth = min(frame + 5, 20)
        depths = np.arange(max_depth)
        
        # φ-螺旋参数
        phi_angles = depths * 2 * np.pi / self.system.phi
        phi_radii = self.system.phi ** (depths / 5)
        
        xs = phi_radii * np.cos(phi_angles)
        ys = phi_radii * np.sin(phi_angles)
        zs = depths
        
        # 颜色渐变
        colors = plt.cm.plasma(depths / max(depths.max(), 1))
        
        # 绘制螺旋
        ax.scatter(xs, ys, zs, c=colors, s=100, alpha=0.8)
        
        # 连接线
        if len(xs) > 1:
            ax.plot(xs, ys, zs, 'blue', alpha=0.6, linewidth=2)
        
        # 当前生长点
        if frame < len(xs):
            ax.scatter([xs[frame]], [ys[frame]], [zs[frame]], 
                      c='red', s=200, alpha=1.0, marker='*')
        
        # φ比例标注
        if frame > 0 and frame < len(depths):
            current_ratio = phi_radii[frame] / (phi_radii[frame-1] if frame > 0 else 1)
            ax.text(xs[frame], ys[frame], zs[frame] + 1, 
                   f'φ≈{current_ratio:.3f}', fontsize=10)
        
        ax.set_xlabel('X (φ-缩放)')
        ax.set_ylabel('Y (φ-缩放)')
        ax.set_zlabel('递归深度')
    
    def visualize_phase_transitions(self):
        """相变点可视化"""
        ax = self.axes['phase_transitions']
        ax.clear()
        ax.set_title('相变点分析', fontsize=14)
        
        transitions = self.system.find_phase_transitions()
        
        if transitions:
            depths = [t['depth'] for t in transitions]
            entropy_jumps = [t['entropy_jump'] for t in transitions]
            complexity_jumps = [t['complexity_jump'] for t in transitions]
            
            # 散点图
            scatter = ax.scatter(depths, entropy_jumps, s=complexity_jumps, 
                               c=range(len(transitions)), cmap='rainbow', 
                               alpha=0.7, edgecolors='black')
            
            # 标注
            for i, t in enumerate(transitions):
                ax.annotate(f"L{t['from_level']}→L{t['to_level']}", 
                           (t['depth'], t['entropy_jump']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlabel('递归深度')
            ax.set_ylabel('熵跳跃')
            
            # 颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('相变序号')
        
        ax.grid(True, alpha=0.3)
    
    def visualize_info_flow_convergence(self, frame: int):
        """信息流收敛可视化"""
        ax = self.axes['info_flow']
        ax.clear()
        ax.set_title(f'信息流收敛 (t={frame})', fontsize=14)
        
        # 创建信息流网络
        levels = self.system.hierarchy_levels
        
        for level in range(min(5, len(levels))):
            if level not in levels or level + 1 not in levels:
                continue
            
            current_states = levels[level]
            next_states = levels[level + 1]
            
            # 计算信息流
            for i, curr_state in enumerate(current_states):
                for j, next_state in enumerate(next_states):
                    # 信息流强度（简化计算）
                    flow_strength = 1.0 / (1 + abs(i - j))
                    
                    if flow_strength > 0.1:  # 只显示强流
                        # 起点
                        x1 = level * 2
                        y1 = i - len(current_states) / 2
                        z1 = curr_state['entropy']
                        
                        # 终点
                        x2 = (level + 1) * 2
                        y2 = j - len(next_states) / 2
                        z2 = next_state['entropy']
                        
                        # 绘制流线
                        alpha = min(1.0, flow_strength * 2)
                        linewidth = max(0.5, flow_strength * 3)
                        
                        ax.plot([x1, x2], [y1, y2], [z1, z2], 
                               'blue', alpha=alpha, linewidth=linewidth)
            
            # 绘制层级节点
            for i, state in enumerate(current_states):
                x = level * 2
                y = i - len(current_states) / 2
                z = state['entropy']
                
                color = 'green' if state['valid'] else 'red'
                ax.scatter([x], [y], [z], c=color, s=50, alpha=0.8)
        
        # 收敛点（简化）
        if frame > 10:
            ax.scatter([10], [0], [5], c='red', s=200, marker='*', 
                      label='收敛点')
            ax.legend()
        
        ax.set_xlabel('层级')
        ax.set_ylabel('状态索引')
        ax.set_zlabel('熵')
    
    def visualize_irreversibility(self):
        """不可逆性演示"""
        ax = self.axes['irreversibility']
        ax.clear()
        ax.set_title('层级转换的不可逆性', fontsize=14)
        
        # 生成转换序列
        states = self.system.calculate_recursive_depth()[:15]
        
        times = [s['depth'] for s in states]
        levels = [s['level'] for s in states]
        entropies = [s['entropy'] for s in states]
        
        # 正向序列
        ax.plot(times, levels, 'g-', linewidth=3, marker='o', 
               markersize=6, label='正向转换（允许）')
        
        # 尝试反向序列
        reverse_levels = levels[::-1]
        ax.plot(times, reverse_levels, 'r--', linewidth=2, marker='x', 
               markersize=6, alpha=0.7, label='反向转换（禁止）')
        
        # 标注不可逆点
        for i, (t, l, rev_l) in enumerate(zip(times, levels, reverse_levels)):
            if l != rev_l and i > 0:
                ax.annotate('不可逆!', (t, rev_l), xytext=(t+0.5, rev_l+0.5),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           color='red', fontweight='bold')
        
        ax.set_xlabel('时间步')
        ax.set_ylabel('层级')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def visualize_level_statistics(self):
        """层级分布统计"""
        ax = self.axes['level_stats']
        ax.clear()
        ax.set_title('层级分布统计', fontsize=14)
        
        levels = self.system.hierarchy_levels
        
        # 统计每层的状态数
        level_counts = {}
        valid_counts = {}
        
        for level, states in levels.items():
            level_counts[level] = len(states)
            valid_counts[level] = sum(1 for s in states if s['valid'])
        
        # 绘制柱状图
        level_nums = list(level_counts.keys())
        total_counts = list(level_counts.values())
        valid_count_vals = [valid_counts.get(l, 0) for l in level_nums]
        invalid_counts = [total - valid for total, valid in 
                         zip(total_counts, valid_count_vals)]
        
        width = 0.6
        ax.bar(level_nums, valid_count_vals, width, label='有效状态', 
               color='green', alpha=0.7)
        ax.bar(level_nums, invalid_counts, width, bottom=valid_count_vals,
               label='违反No-11', color='red', alpha=0.7)
        
        # Fibonacci期望线
        fib_expected = []
        for level in level_nums:
            if level < len(self.system.fibonacci):
                fib_expected.append(self.system.fibonacci[level] / 10)  # 缩放
            else:
                fib_expected.append(0)
        
        ax.plot(level_nums, fib_expected, 'b--', linewidth=2, 
               marker='s', label='Fibonacci期望')
        
        ax.set_xlabel('层级')
        ax.set_ylabel('状态数')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_animated_demo(self):
        """创建动画演示"""
        print("启动T0-11递归层级3D动画演示...")
        
        self.setup_3d_dashboard()
        
        def animate(frame):
            # 清理并重绘所有子图
            self.visualize_3d_hierarchy(frame)
            self.visualize_complexity_surface()
            self.visualize_phi_scaling_animation(frame)
            self.visualize_phase_transitions()
            self.visualize_info_flow_convergence(frame)
            self.visualize_irreversibility()
            self.visualize_level_statistics()
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, animate, frames=30, interval=1000, blit=False, repeat=True
        )
        
        plt.show()
        return anim
    
    def create_static_analysis(self):
        """创建静态分析图表"""
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('T0-11递归层级理论 - 静态分析', fontsize=16, fontweight='bold')
        
        # 创建子图
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. 层级复杂性分析
        ax1 = fig.add_subplot(gs[0, :])
        states = self.system.calculate_recursive_depth()
        depths = [s['depth'] for s in states]
        complexities = [s['complexity'] for s in states]
        entropies = [s['entropy'] for s in states]
        
        ax1.semilogy(depths, complexities, 'b-', linewidth=2, marker='o', 
                    label='实际复杂性')
        
        # φ增长理论线
        theoretical = [self.system.phi ** d for d in depths]
        ax1.semilogy(depths, theoretical, 'r--', linewidth=2, 
                    label=f'φ^d 理论增长')
        
        ax1.set_title('复杂性增长：实际 vs 理论', fontsize=14)
        ax1.set_xlabel('递归深度')
        ax1.set_ylabel('复杂性 (对数尺度)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 熵产生率分析
        ax2 = fig.add_subplot(gs[1, 0])
        if len(entropies) > 1:
            entropy_rates = np.gradient(entropies)
            ax2.plot(depths[1:], entropy_rates[1:], 'g-', linewidth=2, marker='s')
            ax2.axhline(y=math.log2(self.system.phi), color='red', linestyle='--', 
                       label=f'log₂(φ) = {math.log2(self.system.phi):.3f}')
            ax2.set_title('熵产生率')
            ax2.set_xlabel('递归深度')
            ax2.set_ylabel('dH/dd')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 层级跳跃统计
        ax3 = fig.add_subplot(gs[1, 1])
        transitions = self.system.find_phase_transitions()
        if transitions:
            jump_sizes = [t['entropy_jump'] for t in transitions]
            ax3.hist(jump_sizes, bins=10, alpha=0.7, color='purple', edgecolor='black')
            ax3.set_title('层级跳跃分布')
            ax3.set_xlabel('熵跳跃大小')
            ax3.set_ylabel('频次')
            ax3.grid(True, alpha=0.3)
        
        # 4. φ收敛精度
        ax4 = fig.add_subplot(gs[1, 2])
        phi_ratios = []
        for i in range(2, len(self.system.fibonacci)-1):
            ratio = self.system.fibonacci[i] / self.system.fibonacci[i-1]
            phi_ratios.append(ratio)
        
        phi_errors = [abs(r - self.system.phi) for r in phi_ratios]
        ax4.semilogy(range(len(phi_errors)), phi_errors, 'orange', linewidth=2, marker='d')
        ax4.set_title('φ收敛误差')
        ax4.set_xlabel('Fibonacci索引')
        ax4.set_ylabel('|F_n/F_{n-1} - φ|')
        ax4.grid(True, alpha=0.3)
        
        # 5. No-11约束效果分析
        ax5 = fig.add_subplot(gs[2, 0])
        valid_states = [s for s in states if s['valid']]
        invalid_states = [s for s in states if not s['valid']]
        
        valid_depths = [s['depth'] for s in valid_states]
        invalid_depths = [s['depth'] for s in invalid_states]
        
        ax5.hist([valid_depths, invalid_depths], bins=15, alpha=0.7, 
                label=['有效状态', '违反No-11'], color=['green', 'red'])
        ax5.set_title('No-11约束效果')
        ax5.set_xlabel('递归深度')
        ax5.set_ylabel('状态数')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 信息流汇聚分析
        ax6 = fig.add_subplot(gs[2, 1])
        levels = self.system.generate_hierarchy_levels()
        level_entropies = {}
        
        for level, level_states in levels.items():
            if level_states:
                avg_entropy = np.mean([s['entropy'] for s in level_states])
                level_entropies[level] = avg_entropy
        
        if level_entropies:
            level_nums = list(level_entropies.keys())
            avg_entropies = list(level_entropies.values())
            
            ax6.plot(level_nums, avg_entropies, 'cyan', linewidth=2, marker='o')
            ax6.set_title('层级平均熵')
            ax6.set_xlabel('层级')
            ax6.set_ylabel('平均熵')
            ax6.grid(True, alpha=0.3)
        
        # 7. 时间演化轨迹
        ax7 = fig.add_subplot(gs[2, 2])
        time_points = depths[:20]  # 前20步
        complexity_points = complexities[:20]
        entropy_points = entropies[:20]
        
        # 相空间轨迹
        ax7.plot(complexity_points, entropy_points, 'purple', linewidth=2, marker='o', 
                markersize=4, alpha=0.8)
        
        # 标记起点和终点
        ax7.scatter(complexity_points[0], entropy_points[0], c='green', s=100, 
                   marker='s', label='起点')
        if len(complexity_points) > 1:
            ax7.scatter(complexity_points[-1], entropy_points[-1], c='red', s=100, 
                       marker='*', label='终点')
        
        ax7.set_title('相空间轨迹')
        ax7.set_xlabel('复杂性')
        ax7.set_ylabel('熵')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('T0_11_hierarchy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    """主函数"""
    print("="*60)
    print("T0-11: 递归深度与层级理论 - 3D可视化系统")
    print("="*60)
    print()
    
    # 创建系统
    hierarchy_system = RecursiveHierarchySystem()
    visualizer = Hierarchy3DVisualizer(hierarchy_system)
    
    print("选择可视化模式:")
    print("1. 3D动画演示")
    print("2. 静态分析图表")
    print("3. 完整演示（静态+动画）")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == '1':
        print("启动3D动画演示...")
        anim = visualizer.create_animated_demo()
        
    elif choice == '2':
        print("生成静态分析...")
        fig = visualizer.create_static_analysis()
        
    elif choice == '3':
        print("生成静态分析...")
        fig_static = visualizer.create_static_analysis()
        
        input("按回车键继续到3D动画演示...")
        print("启动3D动画演示...")
        anim = visualizer.create_animated_demo()
        
    else:
        print("无效选择，运行默认演示...")
        anim = visualizer.create_animated_demo()
    
    print()
    print("T0-11演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()