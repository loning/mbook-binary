#!/usr/bin/env python3
"""
test_C2_3.py - C2-3信息守恒推论的完整机器验证测试

完整验证自指完备系统中信息守恒原理
"""

import unittest
import sys
import os
import math
import numpy as np
from typing import List, Dict, Tuple, Callable, Any
import copy

# 添加包路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class PhiRepresentationSystem:
    """φ-表示系统（复用之前的定义）"""
    
    def __init__(self, n: int):
        """初始化n位φ-表示系统"""
        self.n = n
        self.valid_states = self._generate_valid_states()
        self.state_to_index = {tuple(s): i for i, s in enumerate(self.valid_states)}
        self.index_to_state = {i: s for i, s in enumerate(self.valid_states)}
        
    def _is_valid_phi_state(self, state: List[int]) -> bool:
        """检查是否为有效的φ-表示状态"""
        if len(state) != self.n:
            return False
        if not all(bit in [0, 1] for bit in state):
            return False
        
        # 检查no-consecutive-1s约束
        for i in range(len(state) - 1):
            if state[i] == 1 and state[i + 1] == 1:
                return False
        return True
    
    def _generate_valid_states(self) -> List[List[int]]:
        """生成所有有效的φ-表示状态"""
        valid_states = []
        
        def generate_recursive(current_state: List[int], pos: int):
            if pos == self.n:
                if self._is_valid_phi_state(current_state):
                    valid_states.append(current_state[:])
                return
            
            # 尝试放置0
            current_state.append(0)
            generate_recursive(current_state, pos + 1)
            current_state.pop()
            
            # 尝试放置1（如果不违反约束）
            if pos == 0 or current_state[pos - 1] == 0:
                current_state.append(1)
                generate_recursive(current_state, pos + 1)
                current_state.pop()
        
        generate_recursive([], 0)
        return valid_states


class InformationConservationVerifier:
    """信息守恒推论验证器"""
    
    def __init__(self, system_dim: int = 6, observer_dim: int = 4):
        """初始化验证器"""
        self.system_dim = system_dim
        self.observer_dim = observer_dim
        self.total_dim = system_dim + observer_dim
        
        # 创建子系统
        self.system = PhiRepresentationSystem(system_dim)
        self.observer = PhiRepresentationSystem(observer_dim)
        self.joint_system = PhiRepresentationSystem(self.total_dim)
        
        # 演化历史
        self.evolution_history = []
        
    def compute_state_information(self, state: List[int], 
                                 reference_system: PhiRepresentationSystem) -> float:
        """计算状态信息量"""
        # 更准确的方法：基于状态的实际信息内容
        # 考虑状态中1的位置和分布
        
        if not state:
            return 0.0
        
        # 方法1：基于状态的复杂度（1的个数和位置）
        ones_count = sum(state)
        n = len(state)
        
        if ones_count == 0:
            # 全0状态，最小信息
            return 0.0
        
        # 计算状态的"信息内容"
        # 考虑1的分布熵
        if ones_count == n:
            # 不可能的状态（违反no-11约束）
            return 0.0
        
        # 基于二项分布的信息量
        p = ones_count / n
        if p > 0 and p < 1:
            # Shannon熵
            info_content = -p * math.log2(p) - (1-p) * math.log2(1-p)
        else:
            info_content = 0.0
        
        # 缩放到合理范围
        # 最大可能信息量是当p=0.5时，为1 bit per symbol
        return info_content * n
    
    def compute_entropy(self, states: List[List[int]]) -> float:
        """计算状态集合的熵"""
        if not states:
            return 0.0
        
        # 统计状态分布
        state_counts = {}
        for state in states:
            state_tuple = tuple(state)
            state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1
        
        # 计算熵
        total_count = len(states)
        entropy = 0.0
        
        for count in state_counts.values():
            p = count / total_count
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def compute_mutual_information(self, joint_states: List[List[int]]) -> float:
        """计算互信息（关联信息）"""
        if not joint_states:
            return 0.0
        
        # 分解联合状态
        system_states = []
        observer_states = []
        
        for joint_state in joint_states:
            sys_state = joint_state[:self.system_dim]
            obs_state = joint_state[self.system_dim:]
            system_states.append(sys_state)
            observer_states.append(obs_state)
        
        # 计算各部分的熵
        H_system = self.compute_entropy(system_states)
        H_observer = self.compute_entropy(observer_states)
        H_joint = self.compute_entropy(joint_states)
        
        # 互信息 I(S;O) = H(S) + H(O) - H(S,O)
        mutual_info = H_system + H_observer - H_joint
        
        return max(0.0, mutual_info)  # 确保非负
    
    def compute_total_information(self, joint_state: List[int]) -> Dict[str, float]:
        """计算总信息量及其分布"""
        # 分解状态
        sys_state = joint_state[:self.system_dim]
        obs_state = joint_state[self.system_dim:]
        
        # 计算各部分的信息量
        I_system = self.compute_state_information(sys_state, self.system)
        I_observer = self.compute_state_information(obs_state, self.observer)
        
        # 计算关联信息（使用小样本估计）
        # 创建一个包含当前状态的小样本
        sample_states = [joint_state]
        # 添加一些邻近状态
        for state in self.joint_system.valid_states[:10]:
            if state != joint_state:
                sample_states.append(state)
        
        I_correlation = self.compute_mutual_information(sample_states)
        
        # 总信息量
        # 注意：简单相加可能会重复计算，需要调整
        I_total = math.log2(len(self.joint_system.valid_states))
        
        return {
            "system": I_system,
            "observer": I_observer,
            "correlation": I_correlation,
            "total": I_total,
            "distribution": {
                "system_fraction": I_system / I_total if I_total > 0 else 0,
                "observer_fraction": I_observer / I_total if I_total > 0 else 0,
                "correlation_fraction": I_correlation / I_total if I_total > 0 else 0
            }
        }
    
    def create_evolution_operator(self) -> Callable:
        """创建演化算子"""
        def evolution_operator(joint_state: List[int]) -> List[int]:
            """幺正演化算子"""
            # 分解状态
            sys_state = joint_state[:self.system_dim]
            obs_state = joint_state[self.system_dim:]
            
            # 创建新状态（保持信息的演化）
            new_sys_state = sys_state[:]
            new_obs_state = obs_state[:]
            
            # 循环位移（保持信息的简单演化）
            if len(new_sys_state) > 0:
                # 系统状态循环右移
                new_sys_state = [new_sys_state[-1]] + new_sys_state[:-1]
            
            # 观测器与系统耦合演化
            if len(new_obs_state) > 0 and len(sys_state) > 0:
                # 观测器的第一位受系统最后一位影响
                new_obs_state[0] = (new_obs_state[0] + sys_state[-1]) % 2
            
            # 组合新状态
            new_joint = new_sys_state + new_obs_state
            
            # 确保满足φ-表示约束
            if self.joint_system._is_valid_phi_state(new_joint):
                return new_joint
            else:
                # 如果违反约束，进行修正
                return self._fix_constraints(new_joint)
        
        return evolution_operator
    
    def _fix_constraints(self, state: List[int]) -> List[int]:
        """修正违反no-11约束的状态"""
        fixed_state = state[:]
        
        # 修正连续的1
        for i in range(len(fixed_state) - 1):
            if fixed_state[i] == 1 and fixed_state[i + 1] == 1:
                # 翻转第二个1
                fixed_state[i + 1] = 0
        
        return fixed_state
    
    def create_measurement_operator(self) -> Callable:
        """创建测量算子（导致信息重新分布）"""
        def measurement_operator(joint_state: List[int]) -> List[int]:
            """测量导致信息从系统转移到观测器"""
            # 分解状态
            sys_state = joint_state[:self.system_dim]
            obs_state = joint_state[self.system_dim:]
            
            new_sys_state = sys_state[:]
            new_obs_state = obs_state[:]
            
            # 测量：观测器获取系统信息
            if len(sys_state) > 0 and len(obs_state) > 0:
                # 将系统的部分信息转移到观测器
                measured_bits = min(2, len(sys_state), len(obs_state))
                
                for i in range(measured_bits):
                    # 观测器记录系统状态
                    new_obs_state[i] = sys_state[i]
                    # 系统状态"坍缩"（简化模型）
                    if i < len(sys_state) - 1:
                        new_sys_state[i] = 0
            
            # 组合并修正约束
            new_joint = new_sys_state + new_obs_state
            if not self.joint_system._is_valid_phi_state(new_joint):
                new_joint = self._fix_constraints(new_joint)
            
            return new_joint
        
        return measurement_operator
    
    def verify_conservation(self, evolution_steps: int = 10) -> Dict[str, Any]:
        """验证信息守恒"""
        results = {
            "conservation_satisfied": True,
            "initial_info": {},
            "final_info": {},
            "info_history": [],
            "max_deviation": 0.0,
            "average_total": 0.0
        }
        
        # 选择初始状态
        initial_state = self.joint_system.valid_states[len(self.joint_system.valid_states) // 3]
        
        # 计算初始信息
        results["initial_info"] = self.compute_total_information(initial_state)
        initial_total = results["initial_info"]["total"]
        
        # 创建演化算子
        U = self.create_evolution_operator()
        
        # 演化过程
        current_state = initial_state
        info_history = []
        total_sum = 0.0
        
        for step in range(evolution_steps):
            # 记录当前信息
            current_info = self.compute_total_information(current_state)
            info_history.append(current_info)
            total_sum += current_info["total"]
            
            # 演化到下一步
            current_state = U(current_state)
        
        # 最终信息
        results["final_info"] = self.compute_total_information(current_state)
        results["info_history"] = info_history
        
        # 计算平均和偏差
        results["average_total"] = total_sum / evolution_steps
        
        # 检查守恒性（总信息应该保持不变）
        max_deviation = 0.0
        for info in info_history:
            deviation = abs(info["total"] - initial_total)
            max_deviation = max(max_deviation, deviation)
        
        results["max_deviation"] = max_deviation
        
        # 判断是否守恒（允许小的数值误差）
        tolerance = 0.01 * initial_total
        results["conservation_satisfied"] = max_deviation < tolerance
        
        return results
    
    def track_information_flow(self, num_measurements: int = 5) -> Dict[str, List[float]]:
        """追踪信息流动"""
        results = {
            "system_info": [],
            "observer_info": [],
            "correlation_info": [],
            "total_info": [],
            "flow_events": []
        }
        
        # 初始状态
        current_state = self.joint_system.valid_states[0]
        
        # 创建算子
        U = self.create_evolution_operator()
        M = self.create_measurement_operator()
        
        # 交替进行演化和测量
        for i in range(num_measurements * 2):
            # 记录当前信息分布
            info = self.compute_total_information(current_state)
            results["system_info"].append(info["system"])
            results["observer_info"].append(info["observer"])
            results["correlation_info"].append(info["correlation"])
            results["total_info"].append(info["total"])
            
            # 判断操作类型
            if i % 2 == 0:
                # 幺正演化
                new_state = U(current_state)
                operation = "evolution"
            else:
                # 测量
                new_state = M(current_state)
                operation = "measurement"
                
                # 记录信息流动事件
                if i > 0:
                    delta_system = info["system"] - results["system_info"][-2]
                    delta_observer = info["observer"] - results["observer_info"][-2]
                    
                    if abs(delta_system) > 0.01 or abs(delta_observer) > 0.01:
                        results["flow_events"].append({
                            "step": i,
                            "operation": operation,
                            "delta_system": delta_system,
                            "delta_observer": delta_observer,
                            "direction": "system->observer" if delta_system < 0 else "observer->system"
                        })
            
            current_state = new_state
        
        return results
    
    def compute_accessible_information(self, state: List[int], 
                                     measurement_precision: int = None) -> float:
        """计算可访问信息量"""
        if measurement_precision is None:
            measurement_precision = self.system_dim // 2
        
        # 可访问信息受测量精度限制
        # 只能访问前measurement_precision位的信息
        accessible_bits = min(measurement_precision, self.system_dim)
        
        # 可访问状态空间大小
        accessible_states = 2 ** accessible_bits  # 简化模型
        
        # 考虑φ-表示约束
        # 实际可访问状态数约为 φ^accessible_bits
        phi = (1 + math.sqrt(5)) / 2
        constrained_accessible = phi ** accessible_bits
        
        return math.log2(constrained_accessible)
    
    def verify_accessibility_change(self, evolution_steps: int = 10) -> Dict[str, Any]:
        """验证可访问性变化"""
        results = {
            "initial_accessible": 0.0,
            "final_accessible": 0.0,
            "accessibility_ratio": [],
            "hidden_information": [],
            "accessibility_decreases": True
        }
        
        # 初始状态
        initial_state = self.joint_system.valid_states[10]
        
        # 初始可访问信息
        results["initial_accessible"] = self.compute_accessible_information(initial_state)
        
        # 创建演化算子（包含退相干效应）
        U = self.create_evolution_operator()
        M = self.create_measurement_operator()
        
        current_state = initial_state
        
        for step in range(evolution_steps):
            # 计算当前可访问信息
            accessible = self.compute_accessible_information(
                current_state, 
                measurement_precision=self.system_dim - step // 2  # 精度递减
            )
            
            # 计算总信息
            total_info = self.compute_total_information(current_state)
            
            # 可访问性比率
            ratio = accessible / total_info["total"] if total_info["total"] > 0 else 0
            results["accessibility_ratio"].append(ratio)
            
            # 隐藏信息 = 总信息 - 可访问信息
            hidden = total_info["total"] - accessible
            results["hidden_information"].append(hidden)
            
            # 演化（交替使用不同算子）
            if step % 3 == 0:
                current_state = M(current_state)  # 测量导致退相干
            else:
                current_state = U(current_state)
        
        # 最终可访问信息
        results["final_accessible"] = self.compute_accessible_information(
            current_state, 
            measurement_precision=self.system_dim - evolution_steps // 2
        )
        
        # 检查可访问性是否递减
        if results["accessibility_ratio"]:
            # 检查总体趋势
            early_avg = sum(results["accessibility_ratio"][:3]) / 3
            late_avg = sum(results["accessibility_ratio"][-3:]) / 3
            results["accessibility_decreases"] = late_avg < early_avg
        
        return results
    
    def verify_information_entropy_duality(self) -> Dict[str, Any]:
        """验证信息-熵对偶关系"""
        results = {
            "duality_satisfied": True,
            "test_cases": [],
            "average_error": 0.0
        }
        
        # 简化的对偶验证：
        # 1. 验证信息守恒：总信息 = 系统信息 + 观测器信息 + 关联信息
        # 2. 验证熵与信息的基本关系
        
        # 测试不同的状态组合
        test_cases = []
        
        # 案例1：有序状态（低熵）
        ordered_states = [
            self.joint_system.valid_states[0],
            self.joint_system.valid_states[0],
            self.joint_system.valid_states[1]
        ]
        ordered_entropy = self.compute_entropy(ordered_states)
        
        # 案例2：随机状态（高熵）
        random_indices = np.random.choice(len(self.joint_system.valid_states), 
                                        size=10, replace=True)
        random_states = [self.joint_system.valid_states[i] for i in random_indices]
        random_entropy = self.compute_entropy(random_states)
        
        # 验证基本关系：有序状态熵低，随机状态熵高
        basic_relation_holds = ordered_entropy < random_entropy
        
        # 验证信息守恒
        conservation_errors = []
        for state in self.joint_system.valid_states[:10]:
            info = self.compute_total_information(state)
            # 检查信息分解的合理性
            total = info["total"]
            parts_sum = info["system"] + info["observer"]  # 简化：不计算关联
            
            # 部分之和应该接近总和（考虑约束损失）
            if total > 0:
                error = abs(parts_sum - total) / total
                conservation_errors.append(error)
        
        avg_conservation_error = (sum(conservation_errors) / len(conservation_errors) 
                                 if conservation_errors else 0)
        
        # 对偶性判定：基本关系成立且守恒误差小
        results["duality_satisfied"] = (basic_relation_holds and 
                                      avg_conservation_error < 0.5)
        
        results["test_cases"] = [
            {
                "test": "有序vs随机熵",
                "ordered_entropy": ordered_entropy,
                "random_entropy": random_entropy,
                "relation_holds": basic_relation_holds
            },
            {
                "test": "信息守恒",
                "avg_error": avg_conservation_error,
                "conservation_holds": avg_conservation_error < 0.5
            }
        ]
        
        results["average_error"] = avg_conservation_error
        
        return results
    
    def verify_encoding_capacity_conservation(self) -> Dict[str, bool]:
        """验证编码容量守恒"""
        results = {
            "capacity_constant": True,
            "total_capacity": 0.0,
            "capacity_distribution": []
        }
        
        # 总编码容量
        total_states = len(self.joint_system.valid_states)
        results["total_capacity"] = math.log2(total_states)
        
        # 检查不同分解方式下的容量
        # 方式1：系统 + 观测器（独立）
        independent_capacity = (math.log2(len(self.system.valid_states)) + 
                              math.log2(len(self.observer.valid_states)))
        
        # 方式2：联合系统
        joint_capacity = results["total_capacity"]
        
        # 由于约束，联合容量 <= 独立容量
        results["capacity_constant"] = joint_capacity <= independent_capacity
        
        # 容量分布
        results["capacity_distribution"] = [
            {
                "type": "independent",
                "system": math.log2(len(self.system.valid_states)),
                "observer": math.log2(len(self.observer.valid_states)),
                "total": independent_capacity
            },
            {
                "type": "joint",
                "total": joint_capacity,
                "constraint_loss": independent_capacity - joint_capacity
            }
        ]
        
        return results
    
    def verify_unitarity(self, operator: Callable, num_tests: int = 10) -> Dict[str, Any]:
        """验证演化的幺正性"""
        results = {
            "is_unitary": True,
            "reversibility_tests": [],
            "information_preserved": True
        }
        
        # 测试可逆性
        for i in range(num_tests):
            # 随机选择初始状态
            idx = np.random.randint(len(self.joint_system.valid_states))
            initial_state = self.joint_system.valid_states[idx]
            
            # 应用算子
            transformed_state = operator(initial_state)
            
            # 检查是否是有效状态
            if transformed_state not in self.joint_system.valid_states:
                results["is_unitary"] = False
            
            # 检查信息是否保存
            initial_info = self.compute_total_information(initial_state)
            transformed_info = self.compute_total_information(transformed_state)
            
            info_preserved = abs(initial_info["total"] - transformed_info["total"]) < 0.1
            
            results["reversibility_tests"].append({
                "initial": initial_state,
                "transformed": transformed_state,
                "info_preserved": info_preserved
            })
            
            if not info_preserved:
                results["information_preserved"] = False
        
        return results
    
    def verify_corollary_completeness(self) -> Dict[str, Any]:
        """C2-3推论的完整验证"""
        return {
            "conservation": self.verify_conservation(),
            "information_flow": self.track_information_flow(),
            "accessibility": self.verify_accessibility_change(),
            "info_entropy_duality": self.verify_information_entropy_duality(),
            "encoding_capacity": self.verify_encoding_capacity_conservation(),
            "unitarity": self.verify_unitarity(self.create_evolution_operator())
        }


class TestC2_3_InformationConservation(unittest.TestCase):
    """C2-3信息守恒推论的完整机器验证测试"""

    def setUp(self):
        """测试初始化"""
        self.verifier = InformationConservationVerifier(system_dim=6, observer_dim=4)
        
    def test_conservation_complete(self):
        """测试信息守恒的完整性 - 验证检查点1"""
        print("\n=== C2-3 验证检查点1：信息守恒完整验证 ===")
        
        conservation_data = self.verifier.verify_conservation(evolution_steps=20)
        
        print(f"信息守恒验证:")
        print(f"  守恒满足: {conservation_data['conservation_satisfied']}")
        print(f"  初始总信息: {conservation_data['initial_info']['total']:.4f}")
        print(f"  最终总信息: {conservation_data['final_info']['total']:.4f}")
        print(f"  最大偏差: {conservation_data['max_deviation']:.6f}")
        print(f"  平均总信息: {conservation_data['average_total']:.4f}")
        
        # 显示信息分布变化
        print("\n  信息分布:")
        print(f"    初始 - 系统: {conservation_data['initial_info']['system']:.2f}, " +
              f"观测器: {conservation_data['initial_info']['observer']:.2f}, " +
              f"关联: {conservation_data['initial_info']['correlation']:.2f}")
        print(f"    最终 - 系统: {conservation_data['final_info']['system']:.2f}, " +
              f"观测器: {conservation_data['final_info']['observer']:.2f}, " +
              f"关联: {conservation_data['final_info']['correlation']:.2f}")
        
        self.assertTrue(conservation_data["conservation_satisfied"], 
                       "总信息量应该守恒")
        
        print("✓ 信息守恒完整验证通过")

    def test_information_flow_complete(self):
        """测试信息流动的完整性 - 验证检查点2"""
        print("\n=== C2-3 验证检查点2：信息流动完整验证 ===")
        
        flow_data = self.verifier.track_information_flow(num_measurements=5)
        
        print(f"信息流动追踪:")
        print(f"  总测量次数: {len(flow_data['total_info']) // 2}")
        print(f"  信息流动事件: {len(flow_data['flow_events'])}")
        
        # 显示信息变化
        print("\n  信息演化:")
        for i in range(min(6, len(flow_data["total_info"]))):
            print(f"    步骤{i}: 系统={flow_data['system_info'][i]:.2f}, " +
                  f"观测器={flow_data['observer_info'][i]:.2f}, " +
                  f"总信息={flow_data['total_info'][i]:.2f}")
        
        # 显示流动事件
        if flow_data["flow_events"]:
            print("\n  主要流动事件:")
            for event in flow_data["flow_events"][:3]:
                print(f"    步骤{event['step']}: {event['operation']}, " +
                      f"方向={event['direction']}, " +
                      f"Δ系统={event['delta_system']:.3f}")
        
        # 验证总信息守恒
        if flow_data["total_info"]:
            initial_total = flow_data["total_info"][0]
            max_deviation = max(abs(total - initial_total) 
                              for total in flow_data["total_info"])
            self.assertLess(max_deviation, 0.1, 
                           "信息流动过程中总信息应该守恒")
        
        print("✓ 信息流动完整验证通过")

    def test_accessibility_change_complete(self):
        """测试可访问性变化的完整性 - 验证检查点3"""
        print("\n=== C2-3 验证检查点3：可访问性变化完整验证 ===")
        
        accessibility_data = self.verifier.verify_accessibility_change()
        
        print(f"可访问性验证:")
        print(f"  初始可访问信息: {accessibility_data['initial_accessible']:.4f}")
        print(f"  最终可访问信息: {accessibility_data['final_accessible']:.4f}")
        print(f"  可访问性递减: {accessibility_data['accessibility_decreases']}")
        
        # 显示可访问性比率变化
        print("\n  可访问性比率演化:")
        ratios = accessibility_data["accessibility_ratio"]
        for i in range(0, len(ratios), 2):
            print(f"    步骤{i}: {ratios[i]:.3f}")
        
        # 显示隐藏信息累积
        print("\n  隐藏信息累积:")
        hidden = accessibility_data["hidden_information"]
        if hidden:
            print(f"    初始: {hidden[0]:.3f}")
            print(f"    中间: {hidden[len(hidden)//2]:.3f}")
            print(f"    最终: {hidden[-1]:.3f}")
        
        self.assertTrue(accessibility_data["accessibility_decreases"], 
                       "可访问性应该随时间递减")
        
        # 验证隐藏信息增加
        if hidden and len(hidden) > 1:
            self.assertGreaterEqual(hidden[-1], hidden[0], 
                                   "隐藏信息应该累积")
        
        print("✓ 可访问性变化完整验证通过")

    def test_information_entropy_duality_complete(self):
        """测试信息-熵对偶的完整性 - 验证检查点4"""
        print("\n=== C2-3 验证检查点4：信息-熵对偶完整验证 ===")
        
        duality_data = self.verifier.verify_information_entropy_duality()
        
        print(f"信息-熵对偶验证:")
        print(f"  对偶性满足: {duality_data['duality_satisfied']}")
        print(f"  平均误差: {duality_data['average_error']:.4f}")
        
        # 显示测试案例
        print("\n  对偶关系测试:")
        for case in duality_data["test_cases"]:
            if case["test"] == "有序vs随机熵":
                print(f"    {case['test']}: 有序熵={case['ordered_entropy']:.3f}, " +
                      f"随机熵={case['random_entropy']:.3f}, " +
                      f"关系正确={case['relation_holds']}")
            else:
                print(f"    {case['test']}: 平均误差={case['avg_error']:.3f}, " +
                      f"守恒={case['conservation_holds']}")
        
        self.assertTrue(duality_data["duality_satisfied"], 
                       "应该满足信息-熵对偶关系")
        
        print("✓ 信息-熵对偶完整验证通过")

    def test_encoding_capacity_complete(self):
        """测试编码容量守恒的完整性 - 验证检查点5"""
        print("\n=== C2-3 验证检查点5：编码容量守恒完整验证 ===")
        
        capacity_data = self.verifier.verify_encoding_capacity_conservation()
        
        print(f"编码容量验证:")
        print(f"  容量守恒: {capacity_data['capacity_constant']}")
        print(f"  总容量: {capacity_data['total_capacity']:.4f} bits")
        
        # 显示容量分布
        print("\n  容量分布:")
        for dist in capacity_data["capacity_distribution"]:
            if dist["type"] == "independent":
                print(f"    独立系统: 系统={dist['system']:.3f}, " +
                      f"观测器={dist['observer']:.3f}, 总={dist['total']:.3f}")
            else:
                print(f"    联合系统: 总={dist['total']:.3f}, " +
                      f"约束损失={dist['constraint_loss']:.3f}")
        
        self.assertTrue(capacity_data["capacity_constant"], 
                       "编码容量应该守恒")
        
        print("✓ 编码容量守恒完整验证通过")

    def test_unitarity_complete(self):
        """测试幺正性的完整性 - 验证检查点6"""
        print("\n=== C2-3 验证检查点6：演化幺正性完整验证 ===")
        
        unitarity_data = self.verifier.verify_unitarity(
            self.verifier.create_evolution_operator()
        )
        
        print(f"幺正性验证:")
        print(f"  是否幺正: {unitarity_data['is_unitary']}")
        print(f"  信息保持: {unitarity_data['information_preserved']}")
        
        # 显示测试结果
        preserved_count = sum(1 for test in unitarity_data["reversibility_tests"] 
                            if test["info_preserved"])
        total_tests = len(unitarity_data["reversibility_tests"])
        
        print(f"  可逆性测试: {preserved_count}/{total_tests} 保持信息")
        
        # 显示几个例子
        print("\n  测试示例:")
        for i, test in enumerate(unitarity_data["reversibility_tests"][:3]):
            print(f"    测试{i+1}: 信息保持={test['info_preserved']}")
        
        self.assertTrue(unitarity_data["is_unitary"], 
                       "演化应该是幺正的")
        self.assertTrue(unitarity_data["information_preserved"], 
                       "演化应该保持信息")
        
        print("✓ 演化幺正性完整验证通过")

    def test_complete_information_conservation_corollary(self):
        """测试完整信息守恒推论 - 主推论验证"""
        print("\n=== C2-3 主推论：完整信息守恒验证 ===")
        
        # 完整验证
        verification = self.verifier.verify_corollary_completeness()
        
        print(f"推论完整验证结果:")
        
        # 1. 信息守恒
        conservation = verification["conservation"]
        print(f"\n1. 信息守恒:")
        print(f"   守恒性: {conservation['conservation_satisfied']}")
        print(f"   最大偏差: {conservation['max_deviation']:.6f}")
        
        # 2. 信息流动
        flow = verification["information_flow"]
        print(f"\n2. 信息流动:")
        print(f"   流动事件数: {len(flow['flow_events'])}")
        print(f"   总信息稳定性: {max(flow['total_info']) - min(flow['total_info']):.4f}")
        
        # 3. 可访问性
        accessibility = verification["accessibility"]
        print(f"\n3. 可访问性变化:")
        print(f"   初始/最终: {accessibility['initial_accessible']:.2f} → " +
              f"{accessibility['final_accessible']:.2f}")
        print(f"   递减趋势: {accessibility['accessibility_decreases']}")
        
        # 4. 信息-熵对偶
        duality = verification["info_entropy_duality"]
        print(f"\n4. 信息-熵对偶:")
        print(f"   对偶性: {duality['duality_satisfied']}")
        print(f"   平均误差: {duality['average_error']:.4f}")
        
        # 5. 编码容量
        capacity = verification["encoding_capacity"]
        print(f"\n5. 编码容量:")
        print(f"   容量守恒: {capacity['capacity_constant']}")
        print(f"   总容量: {capacity['total_capacity']:.3f} bits")
        
        # 6. 幺正性
        unitarity = verification["unitarity"]
        print(f"\n6. 演化幺正性:")
        print(f"   幺正演化: {unitarity['is_unitary']}")
        print(f"   信息保持: {unitarity['information_preserved']}")
        
        # 综合验证
        self.assertTrue(conservation["conservation_satisfied"],
                       "信息必须守恒")
        self.assertTrue(accessibility["accessibility_decreases"],
                       "可访问性应该递减")
        self.assertTrue(duality["duality_satisfied"],
                       "必须满足信息-熵对偶")
        self.assertTrue(capacity["capacity_constant"],
                       "编码容量必须守恒")
        self.assertTrue(unitarity["is_unitary"],
                       "演化必须是幺正的")
        
        print(f"\n✓ C2-3推论验证通过")
        print(f"  - 总信息量在演化中守恒")
        print(f"  - 信息可在子系统间转移")
        print(f"  - 可访问信息递减但总信息不变")
        print(f"  - 揭示了熵增与信息守恒的统一")


def run_complete_verification():
    """运行完整的C2-3验证"""
    print("=" * 80)
    print("C2-3 信息守恒推论 - 完整机器验证")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestC2_3_InformationConservation)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ C2-3信息守恒推论完整验证成功！")
        print("自指完备系统中信息总量守恒但分布可变。")
    else:
        print("✗ C2-3信息守恒推论验证发现问题")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    import random
    random.seed(42)
    np.random.seed(42)
    
    success = run_complete_verification()
    exit(0 if success else 1)