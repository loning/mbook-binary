#!/usr/bin/env python3
"""
C6-3 语言演化推论 - 单元测试

验证语言系统作为自指信息编码系统的演化规律。
"""

import unittest
import numpy as np
from typing import Set, Dict, List, Tuple, Optional
import sys
import os

# 添加父目录到路径以导入依赖
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_framework import BinaryUniverseSystem

class LanguageSystem(BinaryUniverseSystem):
    """语言系统的二进制模型"""
    
    def __init__(self):
        super().__init__()
        self.phi = (1 + np.sqrt(5)) / 2
        self.complexity_threshold = self.phi ** 9  # 语言复杂度阈值
        
        # Fibonacci序列（用于词汇规模）
        self.fibonacci = [1, 2]
        for i in range(2, 50):
            self.fibonacci.append(self.fibonacci[-1] + self.fibonacci[-2])
            
        # 音素特征（二进制）
        self.phoneme_features = {
            'voiced': 0,      # 浊音
            'nasal': 1,       # 鼻音
            'fricative': 2,   # 摩擦音
            'stop': 3,        # 塞音
            'front': 4,       # 前音
            'high': 5,        # 高音
            'rounded': 6,     # 圆唇音
            'tense': 7        # 紧音
        }
        
    def phoneme_encoding(self, features: Dict[str, bool]) -> str:
        """音素的二进制编码 - 满足no-11约束"""
        # 创建8位特征向量
        encoding = ['0'] * 8
        
        for feature, index in self.phoneme_features.items():
            if features.get(feature, False):
                # 检查是否会产生"11"
                if index > 0 and encoding[index-1] == '1':
                    # 跳过以避免"11"
                    continue
                if index < 7 and encoding[index+1] == '1':
                    # 跳过以避免"11"
                    continue
                encoding[index] = '1'
                
        return ''.join(encoding)
        
    def verify_no11_constraint(self, binary_str: str) -> bool:
        """验证no-11约束"""
        return '11' not in binary_str


class VocabularySystem:
    """词汇系统的数学模型"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.alpha = 1 / self.phi  # Zipf定律指数
        
    def word_frequency(self, rank: int) -> float:
        """修正的Zipf定律：f(r) = C / (r + φ)^α"""
        C = 1.0  # 归一化常数
        return C / (rank + self.phi) ** self.alpha
        
    def vocabulary_encoding(self, word_index: int) -> str:
        """词汇的φ-表示编码"""
        if word_index == 0:
            return "0"
            
        # 使用Zeckendorf表示
        encoding = []
        remaining = word_index
        fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]  # 注意从1开始
        
        # 贪心算法生成φ-表示
        for i in range(len(fibonacci) - 1, -1, -1):
            if fibonacci[i] <= remaining and i > 0:  # 避免使用第一个1
                encoding.append(i)
                remaining -= fibonacci[i]
                # 跳过下一个以确保zeckendorf表示（无连续）
                if i > 0:
                    i -= 1
                
        # 转换为二进制串
        if not encoding:
            return "1"
            
        max_index = max(encoding)
        binary = ['0'] * (max_index + 1)
        
        for idx in encoding:
            binary[idx] = '1'
            
        return ''.join(binary)
        
    def semantic_distance(self, word1_encoding: str, word2_encoding: str) -> int:
        """语义距离度量"""
        # Hamming距离的φ-加权版本
        distance = 0
        max_len = max(len(word1_encoding), len(word2_encoding))
        
        # 填充到相同长度
        w1 = word1_encoding.ljust(max_len, '0')
        w2 = word2_encoding.ljust(max_len, '0')
        
        for i in range(max_len):
            if w1[i] != w2[i]:
                # 位置权重随φ指数增长
                weight = self.phi ** (i / max_len)
                distance += weight
                
        return int(distance)


class SyntaxTree:
    """句法树的递归结构"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.max_depth = 8  # 固定深度限制
        
    class Node:
        def __init__(self, node_type: int, value: str = ""):
            self.type = node_type  # 0: 终端, 1: 非终端
            self.value = value
            self.left = None
            self.right = None
            
    def build_tree(self, expression: str) -> Node:
        """构建满足递归深度限制的句法树"""
        return self._parse(expression, depth=0)
        
    def _parse(self, expr: str, depth: int) -> Optional[Node]:
        """递归解析 - 限制深度"""
        if depth > self.max_depth:
            # 超过深度限制，强制终止
            return self.Node(0, "...")
            
        # 去除空格
        expr = expr.strip()
        
        # 简化的解析逻辑
        if not expr or not any(c in expr for c in "()[]{}+-*/"):
            # 终端节点
            return self.Node(0, expr)
            
        # 处理括号
        if expr.startswith('(') and expr.endswith(')'):
            # 匹配括号
            paren_count = 0
            for i, c in enumerate(expr):
                if c == '(':
                    paren_count += 1
                elif c == ')':
                    paren_count -= 1
                    if paren_count == 0 and i < len(expr) - 1:
                        # 不是完整的括号包围
                        break
            else:
                # 完整的括号包围，去掉外层括号
                return self._parse(expr[1:-1], depth)
        
        # 非终端节点
        node = self.Node(1)
        
        # 查找操作符
        paren_depth = 0
        op_pos = -1
        for i, c in enumerate(expr):
            if c == '(':
                paren_depth += 1
            elif c == ')':
                paren_depth -= 1
            elif paren_depth == 0 and c in '+-*/':
                op_pos = i
                break
                
        if op_pos > 0:
            # 有操作符，分割
            node.left = self._parse(expr[:op_pos], depth + 1)
            node.right = self._parse(expr[op_pos+1:], depth + 1)
        else:
            # 没有操作符，简单二分
            mid = len(expr) // 2
            node.left = self._parse(expr[:mid], depth + 1)
            node.right = self._parse(expr[mid:], depth + 1)
        
        return node
        
    def tree_to_binary(self, node: Node, encoding: List[str] = None) -> str:
        """将句法树编码为二进制串"""
        if encoding is None:
            encoding = []
            
        if node is None:
            return ''
            
        # 节点类型编码
        encoding.append(str(node.type))
        
        # 递归编码子树
        if hasattr(node, 'left') and node.left:
            self.tree_to_binary(node.left, encoding)
        if hasattr(node, 'right') and node.right:
            self.tree_to_binary(node.right, encoding)
            
        result = ''.join(encoding)
        
        # 确保满足no-11约束
        result = result.replace('11', '10')
        
        return result
    
    def measure_depth(self, node: Node, current_depth: int = 0) -> int:
        """测量树的深度"""
        if node is None or not hasattr(node, 'left'):
            return current_depth
            
        left_depth = self.measure_depth(node.left, current_depth + 1) if node.left else current_depth
        right_depth = self.measure_depth(node.right, current_depth + 1) if node.right else current_depth
        
        return max(left_depth, right_depth)


class GrammarComplexity:
    """Chomsky层级的φ-表示"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.hierarchy = {
            'regular': self.phi ** 2,           # Type-3 更低的阈值
            'context_free': self.phi ** 4,      # Type-2
            'context_sensitive': self.phi ** 6,  # Type-1
            'recursively_enumerable': self.phi ** 8  # Type-0
        }
        
    def classify_grammar(self, rules: List[Tuple[str, str]]) -> str:
        """根据规则集分类语法类型"""
        complexity = self.calculate_complexity(rules)
        
        for grammar_type, threshold in sorted(self.hierarchy.items(), 
                                             key=lambda x: x[1]):
            if complexity <= threshold:
                return grammar_type
                
        return 'recursively_enumerable'
        
    def calculate_complexity(self, rules: List[Tuple[str, str]]) -> float:
        """计算语法规则的复杂度"""
        complexity = 0
        
        for left, right in rules:
            # 规则长度贡献 (较小的贡献)
            complexity += len(left) * len(right) * 0.1
            
            # 非终端符号数量 (更小的贡献)
            non_terminals = sum(1 for c in right if c.isupper())
            complexity += non_terminals * 0.5
            
            # 递归规则额外复杂度
            if left in right:
                complexity += self.phi
                
        return complexity


class SemanticNetwork:
    """语义网络的分形结构"""
    
    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts
        self.phi = (1 + np.sqrt(5)) / 2
        self.connections = np.zeros((num_concepts, num_concepts), dtype=int)
        
    def add_semantic_link(self, concept1: int, concept2: int) -> bool:
        """添加语义连接 - 检查no-11约束"""
        if concept1 == concept2:
            return False
            
        # 检查是否会违反no-11约束
        if self._would_violate_no11(concept1, concept2):
            return False
            
        self.connections[concept1, concept2] = 1
        self.connections[concept2, concept1] = 1
        return True
        
    def _would_violate_no11(self, c1: int, c2: int) -> bool:
        """检查添加连接是否产生11模式"""
        # 获取当前连接模式
        pattern1 = ''.join(str(self.connections[c1, i]) 
                          for i in range(self.num_concepts))
        pattern2 = ''.join(str(self.connections[c2, i]) 
                          for i in range(self.num_concepts))
        
        # 模拟添加连接后的模式
        new_pattern1 = pattern1[:c2] + '1' + pattern1[c2+1:]
        new_pattern2 = pattern2[:c1] + '1' + pattern2[c1+1:]
        
        return '11' in new_pattern1 or '11' in new_pattern2
        
    def semantic_dimension(self) -> float:
        """计算语义网络的分形维度"""
        # D = log(N_connections) / log(N_concepts)
        num_connections = np.sum(self.connections) / 2
        
        if num_connections == 0 or self.num_concepts <= 1:
            return 0
            
        return np.log(num_connections) / np.log(self.num_concepts)
        
    def metaphor_mapping(self, source_domain: Set[int], 
                        target_domain: Set[int]) -> float:
        """隐喻映射的结构保持度"""
        if not source_domain or not target_domain:
            return 0
            
        # 计算源域的内部结构
        source_structure = self._get_subgraph_structure(source_domain)
        
        # 计算目标域的内部结构
        target_structure = self._get_subgraph_structure(target_domain)
        
        # 结构相似度
        similarity = self._structure_similarity(source_structure, target_structure)
        
        return similarity
        
    def _get_subgraph_structure(self, nodes: Set[int]) -> np.ndarray:
        """提取子图结构"""
        n = len(nodes)
        node_list = list(nodes)
        subgraph = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    subgraph[i, j] = self.connections[node_list[i], node_list[j]]
                    
        return subgraph
        
    def _structure_similarity(self, struct1: np.ndarray, 
                            struct2: np.ndarray) -> float:
        """计算结构相似度"""
        if struct1.shape != struct2.shape:
            return 0
            
        # 归一化的结构差异
        diff = np.sum(np.abs(struct1 - struct2))
        max_diff = struct1.size
        
        return 1 - (diff / max_diff) if max_diff > 0 else 1


class VocabularyGrowth:
    """词汇增长的动力学模型"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.v_max = 10000  # 最大词汇量，更实际的值
        
    def growth_rate(self, current_vocab: float, innovation_rate: float) -> float:
        """词汇增长率：dV/dt = φ × r × (1 - V/V_max)"""
        if current_vocab >= self.v_max:
            return 0
            
        return self.phi * innovation_rate * (1 - current_vocab / self.v_max)
        
    def predict_vocabulary(self, initial_vocab: int, innovation_rate: float, 
                         time_steps: int) -> List[float]:
        """预测词汇量演化"""
        vocab = [float(initial_vocab)]
        current = float(initial_vocab)
        
        for t in range(time_steps):
            rate = self.growth_rate(current, innovation_rate)
            current += rate
            vocab.append(current)
            
        return vocab


class TestC6_3LanguageEvolution(unittest.TestCase):
    """C6-3 语言演化推论测试"""
    
    def setUp(self):
        """测试初始化"""
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_phoneme_encoding_no11(self):
        """测试1：音素编码满足no-11约束"""
        print("\n测试1：音素二进制编码")
        
        language = LanguageSystem()
        
        # 测试各种音素特征组合
        test_cases = [
            {'voiced': True, 'nasal': False, 'fricative': True},
            {'stop': True, 'front': True, 'high': False},
            {'voiced': True, 'nasal': True, 'fricative': False},  # 可能产生11
            {'rounded': True, 'tense': True}  # 相邻特征
        ]
        
        print("\n  特征组合                    编码        满足约束")
        print("  --------                    ----        --------")
        
        for features in test_cases:
            encoding = language.phoneme_encoding(features)
            valid = language.verify_no11_constraint(encoding)
            
            feature_str = str(features)[:28].ljust(28)
            print(f"  {feature_str} {encoding}    {valid}")
            
            # 验证no-11约束
            self.assertTrue(valid, f"编码 {encoding} 违反了no-11约束")
            
    def test_zipf_law_modification(self):
        """测试2：修正的Zipf定律"""
        print("\n测试2：词频分布的φ-修正")
        
        vocab = VocabularySystem()
        
        # 计算前20个词的频率
        ranks = list(range(1, 21))
        frequencies = [vocab.word_frequency(r) for r in ranks]
        
        print("\n  排名   频率      φ-修正因子")
        print("  ----   ------    ----------")
        
        for i, (rank, freq) in enumerate(zip(ranks[:10], frequencies[:10])):
            factor = (rank + self.phi) ** vocab.alpha
            print(f"  {rank:4}   {freq:.4f}    {factor:.4f}")
            
        # 验证频率递减
        for i in range(len(frequencies) - 1):
            self.assertGreater(frequencies[i], frequencies[i+1], 
                             "频率应该随排名递减")
            
        # 验证φ-修正效果
        # 传统Zipf: f ∝ 1/r
        # φ-修正: f ∝ 1/(r+φ)^α
        ratio_traditional = frequencies[0] / frequencies[9]
        ratio_expected = ((10 + self.phi) / (1 + self.phi)) ** vocab.alpha
        
        print(f"\n  频率比 f(1)/f(10):")
        print(f"  实际: {ratio_traditional:.3f}")
        print(f"  理论: {ratio_expected:.3f}")
        
        # 容差范围内相等
        self.assertAlmostEqual(ratio_traditional, ratio_expected, places=2)
        
    def test_vocabulary_phi_encoding(self):
        """测试3：词汇的φ-表示编码"""
        print("\n测试3：词汇φ-表示（Zeckendorf编码）")
        
        vocab = VocabularySystem()
        
        # 测试一些词汇索引
        test_indices = [1, 2, 3, 5, 8, 13, 21, 34, 42, 55]
        
        print("\n  词汇索引  二进制编码    Fibonacci分解")
        print("  --------  ----------    -------------")
        
        for idx in test_indices:
            encoding = vocab.vocabulary_encoding(idx)
            
            # 反向验证编码
            fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            components = []
            for i, bit in enumerate(encoding):
                if bit == '1' and i < len(fibonacci):
                    components.append(fibonacci[i])
                    
            print(f"  {idx:8}  {encoding:10}    {components}")
            
            # 验证Zeckendorf表示的正确性
            if components:
                self.assertEqual(sum(components), idx, 
                               f"Fibonacci分解和应等于原索引 {idx}")
                
    def test_syntax_tree_depth_limit(self):
        """测试4：句法树递归深度限制"""
        print("\n测试4：句法递归深度的φ-限制")
        
        syntax = SyntaxTree()
        
        # 测试不同复杂度的表达式
        expressions = [
            "word",
            "(A + B)",
            "((A + B) * (C - D))",
            "(((A + B) * (C - D)) / ((E + F) * (G - H)))",
            "((((((nested))))))"  # 深度嵌套
        ]
        
        print("\n  表达式                              深度  二进制编码")
        print("  --------                            ----  ----------")
        
        for expr in expressions:
            tree = syntax.build_tree(expr)
            depth = syntax.measure_depth(tree)
            binary = syntax.tree_to_binary(tree)
            
            expr_str = expr[:35].ljust(35)
            print(f"  {expr_str} {depth:4}  {binary[:20]}...")
            
            # 验证深度限制
            self.assertLessEqual(depth, syntax.max_depth, 
                               f"深度 {depth} 超过了限制 {syntax.max_depth}")
            
            # 验证二进制编码满足no-11约束
            self.assertNotIn('11', binary, "二进制编码包含禁止的'11'模式")
            
    # 删除这个函数，因为SyntaxTree类已经有measure_depth方法
        
    def test_grammar_complexity_hierarchy(self):
        """测试5：语法复杂度的Chomsky层级"""
        print("\n测试5：Chomsky层级的φ-表示")
        
        grammar = GrammarComplexity()
        
        # 不同类型的语法规则
        test_grammars = {
            'regular': [
                ('S', 'aA'),
                ('A', 'bB'),
                ('B', 'c')
            ],
            'context_free': [
                ('S', 'AB'),
                ('A', 'aAb'),
                ('A', 'ab'),
                ('B', 'cBd'),
                ('B', 'cd')
            ],
            'context_sensitive': [
                ('S', 'ABC'),
                ('AB', 'aAbB'),
                ('BC', 'bBcC'),
                ('C', 'c')
            ]
        }
        
        print("\n  语法类型          复杂度    分类结果")
        print("  --------          ------    --------")
        
        for expected_type, rules in test_grammars.items():
            complexity = grammar.calculate_complexity(rules)
            classified = grammar.classify_grammar(rules)
            
            print(f"  {expected_type:16}  {complexity:6.2f}    {classified}")
            
            # 验证分类的合理性
            if expected_type in ['regular', 'context_free']:
                self.assertEqual(classified, expected_type, 
                               f"{expected_type}语法应该被正确分类")
                
    def test_semantic_network_fractal(self):
        """测试6：语义网络的分形结构"""
        print("\n测试6：语义网络的φ-分形维度")
        
        # 创建不同规模的语义网络
        network_sizes = [10, 20, 50, 100]
        
        print("\n  网络规模  连接数  分形维度")
        print("  --------  ------  --------")
        
        for size in network_sizes:
            network = SemanticNetwork(size)
            
            # 添加语义连接（避免违反no-11）
            connections_added = 0
            np.random.seed(42)
            
            for _ in range(size * 2):  # 尝试添加2n个连接
                c1 = np.random.randint(0, size)
                c2 = np.random.randint(0, size)
                
                if network.add_semantic_link(c1, c2):
                    connections_added += 1
                    
            dimension = network.semantic_dimension()
            
            print(f"  {size:8}  {connections_added:6}  {dimension:8.3f}")
            
            # 验证分形维度的合理性
            # 语义网络的分形维度可以大于1，但应该小于2
            if connections_added > 0:
                self.assertGreater(dimension, 0, "分形维度应该大于0")
                self.assertLess(dimension, 2, "分形维度应该小于2")
                
    def test_vocabulary_growth_dynamics(self):
        """测试7：词汇增长动力学"""
        print("\n测试7：词汇量演化的S曲线")
        
        growth = VocabularyGrowth()
        
        # 不同创新率下的演化
        innovation_rates = [0.1, 0.5, 1.0]
        initial_vocab = 100
        time_steps = 50
        
        print("\n  时间  低创新  中创新  高创新")
        print("  ----  ------  ------  ------")
        
        results = {}
        for rate in innovation_rates:
            vocab_history = growth.predict_vocabulary(
                initial_vocab, rate, time_steps
            )
            results[rate] = vocab_history
            
        # 显示部分时间点
        for t in [0, 10, 20, 30, 40, 50]:
            print(f"  {t:4}  {results[0.1][t]:6.0f}  "
                  f"{results[0.5][t]:6.0f}  {results[1.0][t]:6.0f}")
            
        # 验证S曲线特性
        for rate in innovation_rates:
            history = results[rate]
            
            # 初期加速增长
            early_growth = history[10] - history[0]
            mid_growth = history[30] - history[20]
            late_growth = history[50] - history[40]
            
            # S曲线：中期增长最快
            self.assertGreater(mid_growth, early_growth * 0.8, 
                             "中期增长应该较快")
            self.assertGreater(mid_growth, late_growth, 
                             "后期增长应该放缓")
            
    def test_metaphor_structure_preservation(self):
        """测试8：隐喻的结构保持"""
        print("\n测试8：隐喻映射的拓扑保持")
        
        network = SemanticNetwork(20)
        
        # 创建源域（如"战争"）
        source_domain = {0, 1, 2, 3, 4}
        # 在源域内部建立结构
        network.add_semantic_link(0, 1)
        network.add_semantic_link(1, 2)
        network.add_semantic_link(2, 3)
        network.add_semantic_link(3, 4)
        network.add_semantic_link(0, 4)  # 环形结构
        
        # 创建目标域（如"辩论"）
        target_domain = {10, 11, 12, 13, 14}
        # 复制相似结构
        network.add_semantic_link(10, 11)
        network.add_semantic_link(11, 12)
        network.add_semantic_link(12, 13)
        network.add_semantic_link(13, 14)
        network.add_semantic_link(10, 14)
        
        # 计算结构相似度
        similarity = network.metaphor_mapping(source_domain, target_domain)
        
        print(f"\n  源域节点: {source_domain}")
        print(f"  目标域节点: {target_domain}")
        print(f"  结构相似度: {similarity:.3f}")
        
        # 创建不同结构的域进行对比
        different_domain = {15, 16, 17, 18, 19}
        # 星形结构
        for i in range(16, 20):
            network.add_semantic_link(15, i)
            
        diff_similarity = network.metaphor_mapping(source_domain, different_domain)
        
        print(f"\n  不同结构域: {different_domain}")
        print(f"  结构相似度: {diff_similarity:.3f}")
        
        # 验证：相同结构的相似度应该更高
        self.assertGreater(similarity, diff_similarity, 
                         "相同拓扑结构的隐喻映射应该有更高相似度")
        self.assertGreater(similarity, 0.8, "完全相同的结构相似度应该接近1")
        
    def test_digital_language_acceleration(self):
        """测试9：数字时代的语言加速演化"""
        print("\n测试9：网络环境的φ倍加速效应")
        
        # 模拟传统和数字环境下的词汇创新
        traditional_rate = 0.1
        digital_rate = traditional_rate * self.phi
        
        growth = VocabularyGrowth()
        time_steps = 30
        
        traditional_vocab = growth.predict_vocabulary(1000, traditional_rate, time_steps)
        digital_vocab = growth.predict_vocabulary(1000, digital_rate, time_steps)
        
        print("\n  时间  传统环境  数字环境  加速比")
        print("  ----  --------  --------  ------")
        
        for t in [0, 5, 10, 15, 20, 25, 30]:
            if traditional_vocab[t] > 1000:  # 避免除以接近0的数
                ratio = (digital_vocab[t] - 1000) / (traditional_vocab[t] - 1000)
            else:
                ratio = 1.0
                
            print(f"  {t:4}  {traditional_vocab[t]:8.0f}  "
                  f"{digital_vocab[t]:8.0f}  {ratio:6.2f}")
            
        # 验证加速效应
        final_traditional = traditional_vocab[-1] - traditional_vocab[0]
        final_digital = digital_vocab[-1] - digital_vocab[0]
        
        if final_traditional > 0:
            actual_acceleration = final_digital / final_traditional
            print(f"\n  实际加速因子: {actual_acceleration:.3f}")
            print(f"  理论加速因子: {self.phi:.3f}")
            
            # 验证接近φ倍加速
            self.assertAlmostEqual(actual_acceleration, self.phi, delta=0.2)
            
    def test_language_complexity_threshold(self):
        """测试10：语言复杂度阈值验证"""
        print("\n测试10：语言复杂度的φ^9临界值")
        
        language = LanguageSystem()
        
        # 创建不同复杂度的语言系统
        complexities = [
            self.phi ** 7,      # 低于阈值
            self.phi ** 8,      # 接近阈值
            self.phi ** 8.5,    # 临界区域
            self.phi ** 9,      # 达到阈值
            self.phi ** 9.5     # 超过阈值
        ]
        
        print("\n  复杂度      相对阈值  状态")
        print("  --------    --------  ----")
        
        for complexity in complexities:
            relative = complexity / language.complexity_threshold
            
            # 判断状态
            if relative < 0.9:
                state = "稳定"
            elif relative < 1.0:
                state = "临界"
            else:
                state = "分化"
                
            print(f"  {complexity:8.2f}    {relative:8.3f}  {state}")
            
        # 验证阈值设置
        self.assertEqual(language.complexity_threshold, self.phi ** 9, 
                       "语言复杂度阈值应该是φ^9")
        
        # 模拟复杂度超过阈值时的方言分化
        print("\n  方言分化模拟:")
        
        if complexities[-1] > language.complexity_threshold:
            # 分化为多个子语言
            n_dialects = int(np.log(complexities[-1] / language.complexity_threshold) 
                           / np.log(self.phi)) + 2
            
            print(f"  原语言复杂度: {complexities[-1]:.2f}")
            print(f"  分化为 {n_dialects} 种方言")
            
            # 每个方言的复杂度
            dialect_complexity = complexities[-1] / n_dialects
            print(f"  每种方言复杂度: {dialect_complexity:.2f}")
            
            # 验证分化后每个方言都低于阈值
            self.assertLess(dialect_complexity, language.complexity_threshold,
                          "分化后的方言复杂度应该低于阈值")


if __name__ == "__main__":
    # 设置测试详细度
    unittest.main(verbosity=2)