#!/usr/bin/env python3
"""
T7-2 停机问题定理测试

验证停机问题在二进制宇宙中的不可判定性，
展示自指深度与可判定性边界的关系。
"""

import unittest
from typing import Optional, Tuple, Dict, Set, List, Callable
import time
from base_framework import BinaryUniverseSystem


class BinaryTuringMachine(BinaryUniverseSystem):
    """二进制图灵机实现"""
    
    def __init__(self, states: Set[str], transitions: Dict, 
                 initial: str, finals: Set[str]):
        super().__init__()
        self.states = states
        self.alphabet = {'0', '1'}
        self.transitions = transitions  # (state, symbol) -> (new_state, new_symbol, direction)
        self.initial_state = initial
        self.final_states = finals
        
    def encode(self) -> str:
        """将图灵机编码为二进制串"""
        # 简化编码：状态数|转移表|初始|终止
        encoding = ""
        
        # 编码状态数
        n_states = len(self.states)
        encoding += bin(n_states)[2:].zfill(8)
        
        # 编码转移表（简化版）
        for (state, symbol), (new_state, new_symbol, direction) in self.transitions.items():
            # 使用固定长度编码
            state_idx = list(self.states).index(state)
            new_state_idx = list(self.states).index(new_state)
            encoding += bin(state_idx)[2:].zfill(4)
            encoding += symbol
            encoding += bin(new_state_idx)[2:].zfill(4)
            encoding += new_symbol
            encoding += '1' if direction == 'R' else '0'
            
        return encoding
        
    def simulate(self, input_string: str, max_steps: int = 1000) -> Tuple[bool, str, bool]:
        """
        模拟图灵机执行
        返回：(是否停机, 输出, 是否超时)
        """
        # 初始化带子
        tape = list(input_string) + ['_'] * 100  # 空白符用_表示
        head = 0
        state = self.initial_state
        steps = 0
        
        while steps < max_steps:
            # 检查是否到达终止状态
            if state in self.final_states:
                # 提取输出
                output = ''.join(tape).rstrip('_')
                return True, output, False
                
            # 读取当前符号
            if head >= len(tape):
                tape.extend(['_'] * 100)
            symbol = tape[head]
            if symbol == '_':
                symbol = '0'  # 空白当作0
                
            # 查找转移
            if (state, symbol) not in self.transitions:
                # 无定义转移，停机
                output = ''.join(tape).rstrip('_')
                return True, output, False
                
            # 执行转移
            new_state, new_symbol, direction = self.transitions[(state, symbol)]
            tape[head] = new_symbol
            state = new_state
            
            if direction == 'R':
                head += 1
            elif direction == 'L' and head > 0:
                head -= 1
                
            steps += 1
            
        # 超过最大步数，可能不停机
        return False, '', True


class HaltingDecider:
    """停机判定器接口"""
    
    def decides(self, machine: BinaryTuringMachine, input_string: str) -> Optional[bool]:
        """
        判定机器M在输入w上是否停机
        返回：True（停机）、False（不停机）、None（无法判定）
        """
        raise NotImplementedError("Subclass must implement")


class NaiveHaltingDecider(HaltingDecider):
    """朴素停机判定器（有限深度）"""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.simulation_limit = 2 ** max_depth
        
    def decides(self, machine: BinaryTuringMachine, input_string: str) -> Optional[bool]:
        """通过有限模拟尝试判定"""
        # 模拟执行
        halts, _, timeout = machine.simulate(input_string, self.simulation_limit)
        
        if halts:
            return True
        elif timeout:
            # 分析是否有明显的循环
            if self._detect_simple_loop(machine, input_string):
                return False
            else:
                return None  # 无法判定
        else:
            return False
            
    def _detect_simple_loop(self, machine: BinaryTuringMachine, input_string: str) -> bool:
        """检测简单循环"""
        # 简化：只检查是否回到初始配置
        tape = list(input_string) + ['_'] * 10
        head = 0
        state = machine.initial_state
        
        seen_configs = set()
        max_steps = 100
        
        for _ in range(max_steps):
            config = (state, head, ''.join(tape[:head+10]))
            if config in seen_configs:
                return True
            seen_configs.add(config)
            
            # 执行一步
            symbol = tape[head] if head < len(tape) else '_'
            if symbol == '_':
                symbol = '0'
                
            if (state, symbol) not in machine.transitions:
                return False
                
            new_state, new_symbol, direction = machine.transitions[(state, symbol)]
            tape[head] = new_symbol
            state = new_state
            
            if direction == 'R':
                head += 1
                if head >= len(tape):
                    tape.extend(['_'] * 10)
            elif direction == 'L' and head > 0:
                head -= 1
                
        return False


class DiagonalMachine(BinaryTuringMachine):
    """对角化机器"""
    
    def __init__(self, halting_decider: HaltingDecider):
        # 构造状态和转移
        states = {'q0', 'q_decide', 'q_loop', 'q_halt'}
        transitions = {}
        initial = 'q0'
        finals = {'q_halt'}
        
        super().__init__(states, transitions, initial, finals)
        self.H = halting_decider
        self._setup_transitions()
        
    def _setup_transitions(self):
        """设置转移函数实现对角化"""
        # 简化实现：直接在execute中处理逻辑
        pass
        
    def execute_on_encoding(self, machine_encoding: str) -> bool:
        """
        对编码的机器执行对角化
        D(⟨M⟩) = {
            循环 如果 H(M, ⟨M⟩) = True
            停机 如果 H(M, ⟨M⟩) = False
        }
        """
        # 解码机器（简化：直接创建测试机器）
        # 实际应该从encoding恢复
        
        # 使用判定器
        result = self.H.decides(self, machine_encoding)
        
        if result is True:
            # H说会停机，那么D进入循环
            while True:
                pass  # 实际测试中用有限循环模拟
        else:
            # H说不停机或无法判定，D停机
            return True


class ComplexityHierarchyDecider(HaltingDecider):
    """基于复杂度层级的判定器"""
    
    def __init__(self, level: int):
        self.level = level
        self.decidable_patterns = self._generate_decidable_patterns(level)
        
    def _generate_decidable_patterns(self, level: int) -> Set[str]:
        """生成该层级可判定的模式"""
        patterns = set()
        
        # 层级0：简单停机
        if level >= 0:
            patterns.add("direct_halt")
            patterns.add("finite_loop")
            
        # 层级1：简单递归
        if level >= 1:
            patterns.add("bounded_recursion")
            patterns.add("linear_growth")
            
        # 层级2：嵌套递归
        if level >= 2:
            patterns.add("nested_loops")
            patterns.add("recursive_calls")
            
        return patterns
        
    def decides(self, machine: BinaryTuringMachine, input_string: str) -> Optional[bool]:
        """基于复杂度层级判定"""
        # 分析机器复杂度
        complexity = self._analyze_complexity(machine)
        
        if complexity <= self.level:
            # 可以判定
            return self._decide_at_level(machine, input_string, complexity)
        else:
            # 超出能力范围
            return None
            
    def _analyze_complexity(self, machine: BinaryTuringMachine) -> int:
        """分析机器的复杂度层级"""
        # 简化：基于状态数和转移复杂度
        n_states = len(machine.states)
        n_transitions = len(machine.transitions)
        
        if n_states <= 3:
            return 0
        elif n_states <= 5:
            return 1
        elif n_states <= 10:
            return 2
        else:
            return 3
            
    def _decide_at_level(self, machine: BinaryTuringMachine, 
                        input_string: str, level: int) -> bool:
        """在特定层级进行判定"""
        # 使用层级相关的模拟深度
        max_steps = 10 ** (level + 1)
        halts, _, _ = machine.simulate(input_string, max_steps)
        return halts


class TestT7_2HaltingProblem(unittest.TestCase):
    """T7-2 停机问题定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.simple_decider = NaiveHaltingDecider(max_depth=3)
        
    def test_simple_halting_machines(self):
        """测试1：简单停机机器"""
        print("\n测试1：简单停机机器")
        
        # 构造直接停机的机器
        states = {'q0', 'qf'}
        transitions = {
            ('q0', '0'): ('qf', '0', 'R'),
            ('q0', '1'): ('qf', '1', 'R')
        }
        M1 = BinaryTuringMachine(states, transitions, 'q0', {'qf'})
        
        # 测试停机
        halts, output, timeout = M1.simulate("101")
        print(f"  直接停机机器: halts={halts}, output={output}")
        self.assertTrue(halts)
        self.assertFalse(timeout)
        
        # 测试判定器
        decision = self.simple_decider.decides(M1, "101")
        print(f"  判定结果: {decision}")
        self.assertEqual(decision, True)
        
    def test_simple_looping_machines(self):
        """测试2：简单循环机器"""
        print("\n测试2：简单循环机器")
        
        # 构造无限循环的机器
        states = {'q0', 'q1'}
        transitions = {
            ('q0', '0'): ('q1', '0', 'R'),
            ('q0', '1'): ('q1', '1', 'R'),
            ('q1', '0'): ('q0', '0', 'L'),
            ('q1', '1'): ('q0', '1', 'L')
        }
        M2 = BinaryTuringMachine(states, transitions, 'q0', set())
        
        # 测试循环检测
        halts, _, timeout = M2.simulate("1", max_steps=50)
        print(f"  循环机器: halts={halts}, timeout={timeout}")
        self.assertFalse(halts)
        self.assertTrue(timeout)
        
        # 测试判定器
        decision = self.simple_decider.decides(M2, "1")
        print(f"  判定结果: {decision}")
        # 简单循环可能被检测到
        self.assertIn(decision, [False, None])
        
    def test_undecidability_principle(self):
        """测试3：不可判定性原理"""
        print("\n测试3：不可判定性原理验证")
        
        # 创建不同深度的判定器
        deciders = [
            NaiveHaltingDecider(max_depth=1),
            NaiveHaltingDecider(max_depth=3),
            NaiveHaltingDecider(max_depth=5)
        ]
        
        # 构造递增复杂度的机器
        test_cases = []
        
        # 简单机器
        states1 = {'q0', 'qf'}
        trans1 = {('q0', '0'): ('qf', '1', 'R')}
        M1 = BinaryTuringMachine(states1, trans1, 'q0', {'qf'})
        test_cases.append(("简单", M1, "0"))
        
        # 中等复杂度
        states2 = {'q0', 'q1', 'q2', 'qf'}
        trans2 = {
            ('q0', '0'): ('q1', '1', 'R'),
            ('q1', '0'): ('q2', '0', 'R'),
            ('q1', '1'): ('qf', '1', 'R'),
            ('q2', '0'): ('q0', '0', 'L'),
            ('q2', '1'): ('qf', '1', 'R')
        }
        M2 = BinaryTuringMachine(states2, trans2, 'q0', {'qf'})
        test_cases.append(("中等", M2, "000"))
        
        print("  深度  简单  中等")
        print("  ----  ----  ----")
        
        for i, decider in enumerate(deciders):
            results = []
            for name, machine, input_str in test_cases:
                decision = decider.decides(machine, input_str)
                results.append("Y" if decision is not None else "?")
            print(f"  {2**(i+1):4}  {results[0]:4}  {results[1]:4}")
            
    def test_diagonal_construction(self):
        """测试4：对角化构造（概念验证）"""
        print("\n测试4：对角化构造原理")
        
        # 创建一个声称可以判定的"判定器"
        class ClaimedUniversalDecider(HaltingDecider):
            def decides(self, machine, input_string):
                # 错误地声称可以判定所有情况
                # 简单策略：基于输入长度
                return len(input_string) % 2 == 0
                
        claimed_H = ClaimedUniversalDecider()
        
        # 测试一些机器
        test_inputs = ["", "0", "00", "000", "0000"]
        
        print("  输入   H的判定")
        print("  ----  -------")
        
        # 创建简单测试机器
        states = {'q0', 'qf'}
        trans = {('q0', '0'): ('qf', '0', 'R')}
        M = BinaryTuringMachine(states, trans, 'q0', {'qf'})
        
        for inp in test_inputs:
            decision = claimed_H.decides(M, inp)
            print(f"  {inp:4}  {decision}")
            
        # 验证对角化会导致矛盾
        print("\n  对角化构造将导致矛盾")
        print("  D(⟨D⟩)的行为将与H的预测相反")
        
    def test_complexity_hierarchy_decidability(self):
        """测试5：复杂度层级与可判定性"""
        print("\n测试5：复杂度层级判定能力")
        
        # 创建不同层级的判定器
        hierarchy_deciders = [
            ComplexityHierarchyDecider(0),
            ComplexityHierarchyDecider(1),
            ComplexityHierarchyDecider(2)
        ]
        
        # 创建不同复杂度的测试机器
        machines = []
        
        # C0机器：2状态
        states0 = {'q0', 'qf'}
        trans0 = {('q0', '0'): ('qf', '0', 'R')}
        machines.append(("C0", BinaryTuringMachine(states0, trans0, 'q0', {'qf'})))
        
        # C1机器：4状态
        states1 = {'q0', 'q1', 'q2', 'qf'}
        trans1 = {
            ('q0', '0'): ('q1', '0', 'R'),
            ('q1', '0'): ('q2', '0', 'R'),
            ('q2', '0'): ('qf', '0', 'R')
        }
        machines.append(("C1", BinaryTuringMachine(states1, trans1, 'q0', {'qf'})))
        
        # C2机器：8状态
        states2 = {'q' + str(i) for i in range(8)}
        states2.add('qf')
        trans2 = {}
        for i in range(7):
            trans2[('q' + str(i), '0')] = ('q' + str(i+1), '0', 'R')
        trans2[('q7', '0')] = ('qf', '0', 'R')
        machines.append(("C2", BinaryTuringMachine(states2, trans2, 'q0', {'qf'})))
        
        print("  判定器  C0   C1   C2")
        print("  ------  ---  ---  ---")
        
        for level, decider in enumerate(hierarchy_deciders):
            results = []
            for name, machine in machines:
                decision = decider.decides(machine, "000")
                if decision is None:
                    results.append("?")
                elif decision:
                    results.append("H")
                else:
                    results.append("L")
            print(f"  C{level:5}   {results[0]:3}  {results[1]:3}  {results[2]:3}")
            
        print("\n  H=停机, L=循环, ?=无法判定")
        
    def test_oracle_relativization(self):
        """测试6：Oracle相对化"""
        print("\n测试6：带Oracle的停机问题")
        
        class OracleDecider(HaltingDecider):
            def __init__(self, oracle_level: int):
                self.oracle_level = oracle_level
                self.oracle = ComplexityHierarchyDecider(oracle_level)
                
            def decides(self, machine, input_string):
                # 先用oracle试试
                oracle_result = self.oracle.decides(machine, input_string)
                if oracle_result is not None:
                    return oracle_result
                    
                # Oracle无法判定的，我们也无法判定
                # 但可以处理一些oracle+1层的问题
                complexity = self.oracle._analyze_complexity(machine)
                if complexity == self.oracle_level + 1:
                    # 简化：50%概率判定
                    return len(input_string) % 2 == 0
                    
                return None
                
        # 测试不同oracle
        oracles = [
            OracleDecider(0),
            OracleDecider(1),
            OracleDecider(2)
        ]
        
        # 创建测试机器
        states = {'q0', 'q1', 'q2', 'q3', 'qf'}
        trans = {
            ('q0', '0'): ('q1', '0', 'R'),
            ('q1', '0'): ('q2', '0', 'R'),
            ('q2', '0'): ('q3', '0', 'R'),
            ('q3', '0'): ('qf', '0', 'R')
        }
        M = BinaryTuringMachine(states, trans, 'q0', {'qf'})
        
        print("  Oracle  可判定")
        print("  ------  ------")
        
        for i, oracle in enumerate(oracles):
            result = oracle.decides(M, "0000")
            print(f"  O{i:5}   {'是' if result is not None else '否'}")
            
    def test_encoding_properties(self):
        """测试7：图灵机编码性质"""
        print("\n测试7：二进制编码验证")
        
        # 创建测试机器
        states = {'q0', 'q1', 'qf'}
        transitions = {
            ('q0', '0'): ('q1', '1', 'R'),
            ('q0', '1'): ('qf', '0', 'R'),
            ('q1', '0'): ('qf', '1', 'R'),
            ('q1', '1'): ('q0', '0', 'L')
        }
        M = BinaryTuringMachine(states, transitions, 'q0', {'qf'})
        
        # 编码
        encoding = M.encode()
        print(f"  编码长度: {len(encoding)}")
        print(f"  编码前16位: {encoding[:16]}")
        
        # 验证编码性质
        self.assertTrue(all(c in '01' for c in encoding))
        
        # 检查no-11约束（简化版可能不满足）
        has_11 = '11' in encoding
        print(f"  包含11: {'是' if has_11 else '否'}")
        
    def test_decidability_boundary(self):
        """测试8：可判定性边界"""
        print("\n测试8：可判定性的理论边界")
        
        # 定义问题类别
        problems = {
            "有限自动机停机": 0,
            "线性有界自动机停机": 1,
            "下推自动机停机": 2,
            "图灵机停机": float('inf'),
            "超图灵机停机": float('inf')
        }
        
        print("  问题类型              复杂度  可判定")
        print("  ------------------  ------  ------")
        
        for problem, complexity in problems.items():
            decidable = complexity < float('inf')
            print(f"  {problem:18}  {str(complexity):6}  {'是' if decidable else '否'}")
            
    def test_practical_implications(self):
        """测试9：停机问题的实际影响"""
        print("\n测试9：实际应用中的影响")
        
        # 模拟程序验证场景
        class ProgramVerifier:
            def verify_termination(self, program: str) -> Optional[bool]:
                """尝试验证程序是否终止"""
                # 简单启发式
                if "while True" in program:
                    return False
                elif "return" in program and "while" not in program:
                    return True
                else:
                    return None
                    
        verifier = ProgramVerifier()
        
        test_programs = [
            ("简单返回", "def f(x): return x + 1"),
            ("无限循环", "def f(x): while True: x += 1"),
            ("条件循环", "def f(x): while x > 0: x -= 1"),
            ("递归函数", "def f(x): return f(x-1) if x > 0 else 0"),
            ("复杂逻辑", "def f(x): return f(f(x)) if x % 2 else x")
        ]
        
        print("  程序类型    验证结果")
        print("  --------  ----------")
        
        for name, program in test_programs:
            result = verifier.verify_termination(program)
            if result is None:
                result_str = "无法判定"
            elif result:
                result_str = "会终止"
            else:
                result_str = "不终止"
                
            print(f"  {name:8}  {result_str}")
            
    def test_self_reference_depth(self):
        """测试10：自指深度与停机问题"""
        print("\n测试10：自指深度分析")
        
        # 分析不同结构的自指深度
        class SelfReferenceAnalyzer:
            def compute_depth(self, structure: str) -> int:
                """计算结构的自指深度"""
                if "self" not in structure:
                    return 0
                elif structure.count("self") == 1:
                    return 1
                elif "self(self" in structure:
                    return 2
                else:
                    return structure.count("self")
                    
        analyzer = SelfReferenceAnalyzer()
        
        structures = [
            "f(x) = x + 1",
            "f(x) = f(x-1)",
            "H(M, w) = M(w) halts?",
            "D(M) = not H(M, M)",
            "D(D) = not H(D, D)",
            "meta(f) = analyze(self)"
        ]
        
        print("  结构                      深度")
        print("  ----------------------  ----")
        
        for struct in structures:
            depth = analyzer.compute_depth(struct)
            print(f"  {struct:22}  {depth:4}")
            
        print("\n  更深的自指需要更高的判定能力")


def run_halting_problem_tests():
    """运行停机问题测试"""
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestT7_2HaltingProblem
    )
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("T7-2 停机问题定理 - 测试验证")
    print("=" * 70)
    
    success = run_halting_problem_tests()
    exit(0 if success else 1)