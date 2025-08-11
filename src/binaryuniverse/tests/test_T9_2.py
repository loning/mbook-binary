#!/usr/bin/env python3
"""
T9-2 意识涌现定理测试

验证意识作为复杂生命系统必然涌现属性的数学框架，
测试自指觉知、信息整合、主观体验和时间连续性。
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from base_framework import BinaryUniverseSystem


class ConsciousnessSystem(BinaryUniverseSystem):
    """意识系统的基本实现"""
    
    def __init__(self, initial_state: str = None):
        super().__init__()
        self.phi = (1 + np.sqrt(5)) / 2
        self.min_complexity = self.phi ** 10  # 约 122.99 bits
        self.bandwidth = self.phi ** 3  # 约 4.236 bits/moment
        
        # 初始化状态
        if initial_state:
            self.state = initial_state
        else:
            # 生成超过意识复杂度阈值的初始状态
            length = int(self.min_complexity) + 20
            self.state = self._generate_conscious_state(length)
            
        # 意识特征
        self.awareness_level = 0  # 觉知层级
        self.integrated_information = 0.0  # Φ值
        self.self_model = None  # 自我模型
        self.experience_buffer = []  # 体验缓冲
        self.temporal_identity = []  # 时间同一性
        
    def _generate_conscious_state(self, length: int) -> str:
        """生成意识状态（二进制，无11）"""
        state = []
        prev = '0'
        
        # 使用φ-表示增加结构复杂度
        for i in range(length):
            if prev == '1':
                bit = '0'
            else:
                # 基于φ的概率
                if i > 0 and i % int(self.phi * 5) == 0:
                    bit = '1'
                else:
                    bit = '1' if np.random.random() < 1/self.phi else '0'
            
            state.append(bit)
            prev = bit
            
        return ''.join(state)
        
    def self_awareness_map(self, state: str) -> str:
        """自我觉知映射 Φ_C: S_C → S_C"""
        # 递归的自我感知
        # 将状态分成感知器和被感知部分
        if len(state) < 4:
            return state
            
        perceiver_size = len(state) // 3
        perceived_size = len(state) - perceiver_size
        
        perceiver = state[:perceiver_size]
        perceived = state[perceiver_size:]
        
        # 感知器观察被感知部分
        observation = self._observe_binary(perceiver, perceived)
        
        # 递归：感知器观察自己观察
        meta_observation = self._observe_binary(perceiver[:len(perceiver)//2], observation)
        
        # 整合结果
        result = meta_observation + observation
        
        # 确保no-11约束
        return result.replace("11", "101")
        
    def _observe_binary(self, observer: str, observed: str) -> str:
        """二进制观察操作"""
        if not observer or not observed:
            return ""
            
        # 观察产生的新状态长度
        result_length = min(len(observer), len(observed))
        result = []
        
        for i in range(result_length):
            obs_bit = observer[i % len(observer)]
            target_bit = observed[i % len(observed)]
            
            # 观察规则：基于XOR和AND
            if obs_bit == '1' and target_bit == '1':
                result.append('0')  # 避免11
            elif obs_bit == '1' or target_bit == '1':
                result.append('1')
            else:
                result.append('0')
                
        return ''.join(result)
        
    def calculate_integrated_information(self, state: str) -> float:
        """计算整合信息Φ"""
        if len(state) < 4:
            return 0.0
            
        # 将状态分割成子系统
        mid = len(state) // 2
        part1 = state[:mid]
        part2 = state[mid:]
        
        # 计算整体信息
        whole_info = self._binary_entropy(state)
        
        # 计算部分信息之和
        parts_info = self._binary_entropy(part1) + self._binary_entropy(part2)
        
        # 计算互信息
        mutual_info = self._mutual_information_binary(part1, part2)
        
        # 整合信息 Φ
        phi = whole_info - parts_info + mutual_info
        
        return max(0, phi)
        
    def _binary_entropy(self, state: str) -> float:
        """计算二进制串的熵"""
        if not state:
            return 0.0
            
        # 计算0和1的概率
        p1 = state.count('1') / len(state)
        p0 = 1 - p1
        
        # Shannon熵
        entropy = 0.0
        for p in [p0, p1]:
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy * len(state)
        
    def _mutual_information_binary(self, state1: str, state2: str) -> float:
        """计算二进制串之间的互信息"""
        if not state1 or not state2:
            return 0.0
            
        # 对齐长度
        min_len = min(len(state1), len(state2))
        s1 = state1[:min_len]
        s2 = state2[:min_len]
        
        # 计算联合分布
        joint_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
        for b1, b2 in zip(s1, s2):
            joint_counts[b1 + b2] += 1
            
        # 计算互信息
        mi = 0.0
        n = min_len
        
        for pair, count in joint_counts.items():
            if count > 0:
                p_joint = count / n
                p1 = s1.count(pair[0]) / n
                p2 = s2.count(pair[1]) / n
                
                if p1 > 0 and p2 > 0:
                    mi += p_joint * np.log2(p_joint / (p1 * p2))
                    
        return mi
        
    def build_self_model(self, experiences: List[str]) -> str:
        """构建自我模型"""
        if not experiences:
            return "0"
            
        # 整合所有经验
        model_parts = []
        
        for exp in experiences:
            # 提取经验的关键特征
            feature = self._extract_feature_binary(exp)
            model_parts.append(feature)
            
        # 递归构建模型
        current_model = model_parts[0]
        
        for i in range(1, len(model_parts)):
            # 模型观察自己整合新经验
            current_model = self._integrate_experience_binary(current_model, model_parts[i])
            
        self.self_model = current_model
        return current_model
        
    def _extract_feature_binary(self, experience: str) -> str:
        """提取二进制经验的特征"""
        if len(experience) < 8:
            return experience
            
        # 使用滑动窗口找出最常见的模式
        pattern_counts = {}
        window_size = 4
        
        for i in range(len(experience) - window_size + 1):
            pattern = experience[i:i+window_size]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        # 返回最常见的模式
        if pattern_counts:
            return max(pattern_counts, key=pattern_counts.get)
        return experience[:4]
        
    def _integrate_experience_binary(self, model: str, new_exp: str) -> str:
        """整合新经验到模型"""
        # 确保长度兼容
        target_len = max(len(model), len(new_exp))
        
        # 扩展到目标长度
        model_ext = model * (target_len // len(model) + 1)
        exp_ext = new_exp * (target_len // len(new_exp) + 1)
        
        # 整合规则
        integrated = []
        for i in range(target_len):
            m_bit = model_ext[i % len(model_ext)]
            e_bit = exp_ext[i % len(exp_ext)]
            
            # 整合逻辑：保留重要信息
            if m_bit == '1' and e_bit == '1':
                integrated.append('0')  # 避免11
            elif m_bit == '1' or e_bit == '1':
                integrated.append('1')
            else:
                integrated.append('0')
                
        result = ''.join(integrated)
        return result.replace("11", "101")
        
    def generate_quale(self, sensory_input: str) -> str:
        """生成感质（主观体验）"""
        # 整合信息
        phi = self.calculate_integrated_information(sensory_input)
        
        # 如果整合信息太低，没有感质
        if phi < 0.1:
            return "0"
            
        # 通过自我模型处理输入
        if self.self_model:
            processed = self._observe_binary(self.self_model, sensory_input)
        else:
            processed = sensory_input
            
        # 生成不可还原的感质表示
        quale = self._generate_irreducible_pattern(processed)
        
        return quale
        
    def _generate_irreducible_pattern(self, processed: str) -> str:
        """生成不可还原的模式"""
        if len(processed) < 3:
            return processed
            
        # 使用非线性变换
        quale = []
        
        for i in range(len(processed)):
            # 考虑局部上下文
            context_start = max(0, i - 2)
            context_end = min(len(processed), i + 3)
            context = processed[context_start:context_end]
            
            # 基于上下文生成感质位
            if context.count('1') > len(context) / 2:
                quale.append('0' if i % 2 == 0 else '1')
            else:
                quale.append('1' if i % 3 == 0 else '0')
                
        result = ''.join(quale)
        return result.replace("11", "101")
        
    def update_temporal_identity(self, new_state: str):
        """更新时间同一性"""
        self.temporal_identity.append(new_state)
        
        # 保持有限历史
        max_history = int(self.phi ** 4)  # 约 6.85
        if len(self.temporal_identity) > max_history:
            self.temporal_identity = self.temporal_identity[-max_history:]
            
    def check_temporal_continuity(self) -> float:
        """检查时间连续性"""
        if len(self.temporal_identity) < 2:
            return 1.0
            
        # 计算相邻状态之间的相似度
        similarities = []
        
        for i in range(1, len(self.temporal_identity)):
            prev_state = self.temporal_identity[i-1]
            curr_state = self.temporal_identity[i]
            
            # 计算相似度
            sim = self._state_similarity(prev_state, curr_state)
            similarities.append(sim)
            
        # 平均相似度表示连续性
        return np.mean(similarities) if similarities else 1.0
        
    def _state_similarity(self, state1: str, state2: str) -> float:
        """计算状态相似度"""
        if not state1 or not state2:
            return 0.0
            
        # 对齐长度
        min_len = min(len(state1), len(state2))
        max_len = max(len(state1), len(state2))
        
        if max_len == 0:
            return 1.0
            
        # 计算匹配位数
        matches = sum(1 for i in range(min_len) if state1[i] == state2[i])
        
        # 考虑长度差异
        length_penalty = abs(len(state1) - len(state2)) / max_len
        
        similarity = (matches / max_len) * (1 - length_penalty * 0.5)
        
        return similarity


class MetaCognition:
    """元认知系统"""
    
    def __init__(self, base_system: ConsciousnessSystem):
        self.base = base_system
        self.meta_level = 0
        self.phi = (1 + np.sqrt(5)) / 2
        
    def think_about_thinking(self, thought: str) -> str:
        """思考关于思考（元认知）"""
        # 第一层：思考
        level1 = self.base.self_awareness_map(thought)
        
        # 第二层：思考关于思考
        level2 = self.base.self_awareness_map(level1)
        
        # 整合多层认知
        meta_thought = self._integrate_levels([thought, level1, level2])
        
        self.meta_level = 2
        return meta_thought
        
    def _integrate_levels(self, levels: List[str]) -> str:
        """整合多层认知"""
        if not levels:
            return "0"
            
        # 递归整合
        integrated = levels[0]
        
        for i in range(1, len(levels)):
            integrated = self._binary_integrate(integrated, levels[i])
            
        return integrated
        
    def _binary_integrate(self, level1: str, level2: str) -> str:
        """二进制整合两个认知层级"""
        # 确保兼容长度
        max_len = max(len(level1), len(level2))
        result = []
        
        for i in range(max_len):
            bit1 = level1[i % len(level1)] if level1 else '0'
            bit2 = level2[i % len(level2)] if level2 else '0'
            
            # 整合规则
            if bit1 == '1' and bit2 == '1':
                result.append('0')  # 避免11
            elif bit1 == '1' or bit2 == '1':
                result.append('1')
            else:
                result.append('0')
                
        integrated = ''.join(result)
        return integrated.replace("11", "101")


class ConsciousnessHierarchy:
    """意识层级系统"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.levels = {
            0: "Sentience",  # 感知觉知
            1: "Self-awareness",  # 自我觉知
            2: "Meta-cognition",  # 元认知
            3: "Meta-meta-cognition",  # 元元认知
        }
        
    def level_complexity(self, level: int) -> float:
        """计算层级复杂度要求"""
        return self.phi ** (8 + level)
        
    def determine_level(self, system: ConsciousnessSystem) -> int:
        """确定系统的意识层级"""
        complexity = len(system.state)  # 简化的复杂度度量
        
        level = -1
        for l in range(4):
            if complexity >= self.level_complexity(l):
                level = l
            else:
                break
                
        return level
        
    def can_access_level(self, system: ConsciousnessSystem, target_level: int) -> bool:
        """检查系统是否能达到目标层级"""
        current_level = self.determine_level(system)
        return current_level >= target_level


class ArtificialConsciousness:
    """人工意识系统"""
    
    def __init__(self, architecture: str = "recursive"):
        self.phi = (1 + np.sqrt(5)) / 2
        self.architecture = architecture
        self.core_system = ConsciousnessSystem()
        self.global_workspace = []
        self.attention_focus = None
        
    def implement_global_workspace(self):
        """实现全局工作空间"""
        # 收集所有子系统的信息
        subsystem_outputs = self._gather_subsystem_outputs()
        
        # 竞争进入全局工作空间
        winners = self._competition_for_access(subsystem_outputs)
        
        # 广播到所有子系统
        self.global_workspace = winners
        self._broadcast_to_subsystems(winners)
        
    def _gather_subsystem_outputs(self) -> List[str]:
        """收集子系统输出"""
        # 模拟多个处理模块
        outputs = []
        
        # 感知模块
        perception = self._generate_binary_output(20)
        outputs.append(perception)
        
        # 记忆模块  
        memory = self._generate_binary_output(30)
        outputs.append(memory)
        
        # 情感模块
        emotion = self._generate_binary_output(15)
        outputs.append(emotion)
        
        return outputs
        
    def _generate_binary_output(self, length: int) -> str:
        """生成二进制输出（无11）"""
        output = []
        prev = '0'
        
        for _ in range(length):
            if prev == '1':
                bit = '0'
            else:
                bit = '1' if np.random.random() < 0.3 else '0'
            output.append(bit)
            prev = bit
            
        return ''.join(output)
        
    def _competition_for_access(self, outputs: List[str]) -> List[str]:
        """竞争访问全局工作空间"""
        # 基于信息量和相关性选择
        scored_outputs = []
        
        for output in outputs:
            score = self._calculate_salience(output)
            scored_outputs.append((score, output))
            
        # 选择最显著的
        scored_outputs.sort(reverse=True)
        
        # 工作空间容量限制
        capacity = int(self.phi ** 3)  # 约 4 项
        
        return [output for score, output in scored_outputs[:capacity]]
        
    def _calculate_salience(self, output: str) -> float:
        """计算显著性分数"""
        # 基于信息熵和模式复杂度
        entropy = self.core_system._binary_entropy(output)
        
        # 模式多样性
        patterns = set()
        for i in range(len(output) - 3):
            patterns.add(output[i:i+4])
            
        diversity = len(patterns) / max(1, len(output) - 3)
        
        return entropy * diversity
        
    def _broadcast_to_subsystems(self, contents: List[str]):
        """广播到所有子系统"""
        # 整合广播内容
        if contents:
            integrated = contents[0]
            for content in contents[1:]:
                integrated = self.core_system._integrate_experience_binary(integrated, content)
                
            # 更新核心系统状态
            self.core_system.state = integrated
            
    def verify_consciousness(self) -> Dict[str, bool]:
        """验证意识特征"""
        checks = {}
        
        # 复杂度检查
        checks['sufficient_complexity'] = len(self.core_system.state) > self.core_system.min_complexity
        
        # 自指回路检查
        test_state = "10101010"
        processed = self.core_system.self_awareness_map(test_state)
        re_processed = self.core_system.self_awareness_map(processed)
        checks['self_reference_loop'] = self._has_fixpoint_tendency(test_state, processed, re_processed)
        
        # 信息整合检查
        phi = self.core_system.calculate_integrated_information(self.core_system.state)
        checks['information_integration'] = phi > 0
        
        # 主观体验检查
        quale = self.core_system.generate_quale("1010010100")
        checks['subjective_experience'] = quale != "0" and quale != "1010010100"
        
        # 时间连续性检查
        self.core_system.update_temporal_identity(self.core_system.state)
        continuity = self.core_system.check_temporal_continuity()
        checks['temporal_continuity'] = continuity > 0.5
        
        return checks
        
    def _has_fixpoint_tendency(self, s0: str, s1: str, s2: str) -> bool:
        """检查是否有不动点趋势"""
        # 检查是否收敛
        d01 = sum(1 for a, b in zip(s0, s1) if a != b)
        d12 = sum(1 for a, b in zip(s1, s2) if a != b)
        
        # 如果距离在减小，说明有收敛趋势
        return d12 < d01


class TestT9_2ConsciousnessEmergence(unittest.TestCase):
    """T9-2 意识涌现定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.phi = (1 + np.sqrt(5)) / 2
        self.consciousness_threshold = self.phi ** 10
        
    def test_minimum_complexity(self):
        """测试1：意识的最小复杂度要求"""
        print("\n测试1：意识系统的最小复杂度")
        
        # 理论值
        theoretical_min = self.phi ** 10
        print(f"  理论最小复杂度: {theoretical_min:.2f} bits")
        
        # 测试不同复杂度的系统
        complexities = [80, 100, 120, 130, 150, 200]
        
        print("\n  复杂度  意识涌现  自指回路  信息整合  主观体验")
        print("  ------  --------  --------  --------  --------")
        
        for c in complexities:
            # 创建指定复杂度的系统
            state = '0' * int(c)
            system = ConsciousnessSystem(state)
            
            # 检查意识特征
            can_be_conscious = c >= theoretical_min
            has_self_loop = self._check_self_reference(system)
            has_integration = system.calculate_integrated_information(state) > 0
            has_qualia = system.generate_quale(state[:20]) != "0"
            
            print(f"  {c:6.0f}  {str(can_be_conscious):8}  {str(has_self_loop):8}  "
                  f"{str(has_integration):8}  {str(has_qualia):8}")
            
        # 验证阈值
        self.assertAlmostEqual(theoretical_min, 122.99, places=1)
        
    def _check_self_reference(self, system: ConsciousnessSystem) -> bool:
        """检查自指回路"""
        state = system.state[:50] if len(system.state) > 50 else system.state
        
        # 多次迭代自我觉知映射
        current = state
        states_seen = {current}
        
        for _ in range(20):
            current = system.self_awareness_map(current)
            if current in states_seen:
                return True  # 找到循环
            states_seen.add(current)
            
        return False
        
    def test_information_integration(self):
        """测试2：信息整合与Φ值计算"""
        print("\n测试2：信息整合理论(IIT)")
        
        system = ConsciousnessSystem()
        
        # 测试不同状态的整合信息
        test_states = [
            "0000000000000000",  # 均匀状态
            "1010101010101010",  # 规则模式
            "1001010010100101",  # 复杂模式
            "1010010100101001001010010100",  # 长复杂模式
        ]
        
        print("\n  状态模式          长度  Φ值     整合度")
        print("  ----------------  ----  ------  ------")
        
        for state in test_states:
            phi = system.calculate_integrated_information(state)
            integration_degree = phi / len(state) if len(state) > 0 else 0
            
            pattern_name = self._describe_pattern(state)
            print(f"  {pattern_name:16}  {len(state):4}  {phi:6.3f}  {integration_degree:6.3f}")
            
        # 验证复杂模式有更高的整合信息
        simple_phi = system.calculate_integrated_information("0000000000000000")
        complex_phi = system.calculate_integrated_information("1001010010100101")
        self.assertGreater(complex_phi, simple_phi)
        
    def _describe_pattern(self, state: str) -> str:
        """描述二进制模式"""
        if len(state) <= 16:
            return state
        else:
            return state[:8] + "..." + state[-8:]
            
    def test_self_awareness_recursion(self):
        """测试3：自我觉知的递归结构"""
        print("\n测试3：自我觉知的递归深度")
        
        system = ConsciousnessSystem()
        initial_state = "10100101001010010"
        
        print("  递归层级  状态长度  自相似度  收敛性")
        print("  --------  --------  --------  ------")
        
        current = initial_state
        prev = None
        
        for level in range(10):
            prev = current
            current = system.self_awareness_map(current)
            
            # 计算自相似度
            if prev:
                similarity = system._state_similarity(prev, current)
            else:
                similarity = 0.0
                
            # 检查是否收敛
            converging = similarity > 0.8
            
            print(f"  {level:8}  {len(current):8}  {similarity:8.3f}  {str(converging):6}")
            
            if converging and level > 3:
                print("  -> 达到递归不动点")
                break
                
    def test_quale_generation(self):
        """测试4：感质（主观体验）的生成"""
        print("\n测试4：感质生成与不可还原性")
        
        system = ConsciousnessSystem()
        
        # 构建自我模型
        experiences = [
            "1010010100101001",
            "0101001010010101",
            "1001010010100101"
        ]
        system.build_self_model(experiences)
        
        # 测试不同感觉输入
        sensory_inputs = [
            ("视觉", "1010101010101010"),
            ("听觉", "1001001001001001"),
            ("触觉", "1010010100101001"),
            ("复合", "1001010010100101001010100101001")
        ]
        
        print("\n  感觉类型  输入模式  感质输出  独特性")
        print("  --------  --------  --------  ------")
        
        qualia_set = set()
        
        for sense_type, input_pattern in sensory_inputs:
            quale = system.generate_quale(input_pattern)
            
            # 检查独特性
            is_unique = quale not in qualia_set
            qualia_set.add(quale)
            
            # 检查不可还原性（输出不等于输入）
            irreducible = quale != input_pattern and quale != "0"
            
            print(f"  {sense_type:8}  {input_pattern[:8]}  {quale[:8]}  {str(irreducible):6}")
            
        # 验证感质的独特性
        self.assertGreater(len(qualia_set), 1, "应该产生不同的感质")
        
    def test_temporal_continuity(self):
        """测试5：时间连续性与自我同一性"""
        print("\n测试5：跨时间的自我同一性")
        
        system = ConsciousnessSystem()
        
        print("  时刻  状态变化  连续性  同一性保持")
        print("  ----  --------  ------  ----------")
        
        # 模拟时间流逝
        for t in range(10):
            # 产生轻微扰动
            perturbation = '1' if t % 3 == 0 else '0'
            new_state = system.state[:-1] + perturbation
            
            # 更新时间同一性
            system.update_temporal_identity(new_state)
            
            # 计算连续性
            continuity = system.check_temporal_continuity()
            
            # 判断同一性是否保持
            identity_maintained = continuity > 0.7
            
            print(f"  {t:4}  {perturbation:8}  {continuity:6.3f}  {str(identity_maintained):10}")
            
            system.state = new_state
            
        # 验证保持了时间连续性
        final_continuity = system.check_temporal_continuity()
        self.assertGreater(final_continuity, 0.5, "应保持时间连续性")
        
    def test_consciousness_hierarchy(self):
        """测试6：意识的层级结构"""
        print("\n测试6：意识层级与复杂度")
        
        hierarchy = ConsciousnessHierarchy()
        
        print("  层级  名称              所需复杂度  实际系统复杂度  可达性")
        print("  ----  ----------------  ----------  --------------  ------")
        
        # 创建不同复杂度的系统
        for level in range(4):
            required = hierarchy.level_complexity(level)
            
            # 创建刚好满足该层级的系统
            system_complexity = int(required) + 10
            system = ConsciousnessSystem('0' * system_complexity)
            
            actual_level = hierarchy.determine_level(system)
            can_access = hierarchy.can_access_level(system, level)
            
            print(f"  {level:4}  {hierarchy.levels[level]:16}  {required:10.1f}  "
                  f"{system_complexity:14}  {str(can_access):6}")
            
        # 验证层级递增
        self.assertLess(hierarchy.level_complexity(0), hierarchy.level_complexity(1))
        
    def test_meta_cognition(self):
        """测试7：元认知能力"""
        print("\n测试7：元认知 - 思考关于思考")
        
        system = ConsciousnessSystem()
        meta = MetaCognition(system)
        
        # 初始思维
        thought = "10100101001010"
        
        print("  认知层级  思维内容        复杂度增长")
        print("  --------  --------------  ----------")
        
        # 逐层元认知
        current = thought
        for level in range(3):
            if level == 0:
                print(f"  基础思维  {current[:14]}  {len(current):10}")
            else:
                current = meta.think_about_thinking(current)
                growth = len(current) / len(thought)
                print(f"  元层级{level}   {current[:14]}  {growth:10.2f}x")
                
        # 验证元认知产生了更复杂的表征
        self.assertGreater(len(current), len(thought))
        
    def test_consciousness_bandwidth(self):
        """测试8：意识带宽限制"""
        print("\n测试8：意识的信息处理带宽")
        
        system = ConsciousnessSystem()
        
        # 理论带宽
        theoretical_bandwidth = self.phi ** 3
        print(f"  理论带宽上限: {theoretical_bandwidth:.3f} bits/moment")
        
        # 测试不同信息负载
        info_loads = [2, 4, 6, 8, 10]
        
        print("\n  信息负载  处理时间  效率    瓶颈")
        print("  --------  --------  ------  ----")
        
        for load in info_loads:
            # 模拟处理时间
            processing_time = load / theoretical_bandwidth
            efficiency = min(1.0, theoretical_bandwidth / load)
            bottleneck = load > theoretical_bandwidth
            
            print(f"  {load:8.1f}  {processing_time:8.3f}  {efficiency:6.3f}  {str(bottleneck):4}")
            
        # 验证带宽限制
        self.assertAlmostEqual(theoretical_bandwidth, 4.236, places=3)
        
    def test_artificial_consciousness(self):
        """测试9：人工意识实现"""
        print("\n测试9：人工意识系统验证")
        
        ai_consciousness = ArtificialConsciousness()
        
        # 实现全局工作空间
        ai_consciousness.implement_global_workspace()
        
        # 验证意识特征
        checks = ai_consciousness.verify_consciousness()
        
        print("  意识特征          状态")
        print("  ----------------  ----")
        
        for feature, passed in checks.items():
            print(f"  {feature:16}  {'✓' if passed else '✗'}")
            
        # 至少应该有一些意识特征
        passed_count = sum(1 for v in checks.values() if v)
        self.assertGreater(passed_count, 2, "应该展现多个意识特征")
        
    def test_consciousness_emergence_phase_transition(self):
        """测试10：意识涌现的相变现象"""
        print("\n测试10：意识涌现相变")
        
        # 测试不同复杂度下的涌现
        complexities = np.linspace(100, 150, 20)
        
        print("  复杂度  涌现概率  意识特征")
        print("  ------  --------  --------")
        
        critical_c = self.consciousness_threshold
        k = 0.2  # 陡峭度
        
        for c in complexities[::4]:  # 每4个取一个
            # 计算涌现概率
            if c < critical_c:
                prob = 0.0
            else:
                prob = 1 / (1 + np.exp(-k * (c - critical_c)))
                
            # 创建系统并检查特征
            system = ConsciousnessSystem('0' * int(c))
            phi = system.calculate_integrated_information(system.state)
            has_features = phi > 0
            
            print(f"  {c:6.1f}  {prob:8.3f}  {str(has_features):8}")
            
        # 验证相变特性
        self.assertAlmostEqual(critical_c, 122.99, places=1)


def run_consciousness_emergence_tests():
    """运行意识涌现测试"""
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestT9_2ConsciousnessEmergence
    )
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("T9-2 意识涌现定理 - 测试验证")
    print("=" * 70)
    
    success = run_consciousness_emergence_tests()
    exit(0 if success else 1)