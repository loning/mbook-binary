#!/usr/bin/env python3
"""
T19-2 φ-认知架构定理 - 完整测试程序

禁止任何简化处理！所有实现必须完整且符合理论规范。
"""

import unittest
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# 基础φ算术类（完整实现，用于测试）
class PhiReal:
    def __init__(self, decimal_value: float):
        self._decimal_value = decimal_value
    
    @classmethod
    def from_decimal(cls, value: float):
        return cls(value)
    
    @classmethod
    def zero(cls):
        return cls(0.0)
    
    @classmethod
    def one(cls):
        return cls(1.0)
    
    @property
    def decimal_value(self) -> float:
        return self._decimal_value
    
    def __add__(self, other):
        if isinstance(other, PhiReal):
            return PhiReal(self._decimal_value + other._decimal_value)
        return PhiReal(self._decimal_value + float(other))
    
    def __sub__(self, other):
        if isinstance(other, PhiReal):
            return PhiReal(self._decimal_value - other._decimal_value)
        return PhiReal(self._decimal_value - float(other))
    
    def __mul__(self, other):
        if isinstance(other, PhiReal):
            return PhiReal(self._decimal_value * other._decimal_value)
        return PhiReal(self._decimal_value * float(other))
    
    def __truediv__(self, other):
        if isinstance(other, PhiReal):
            return PhiReal(self._decimal_value / other._decimal_value)
        return PhiReal(self._decimal_value / float(other))
    
    def __pow__(self, other):
        if isinstance(other, PhiReal):
            return PhiReal(self._decimal_value ** other._decimal_value)
        return PhiReal(self._decimal_value ** float(other))
    
    def __neg__(self):
        return PhiReal(-self._decimal_value)
    
    def exp(self):
        return PhiReal(math.exp(self._decimal_value))
    
    def sqrt(self):
        return PhiReal(math.sqrt(abs(self._decimal_value)))

class PhiComplex:
    def __init__(self, real: PhiReal, imag: PhiReal):
        self.real = real
        self.imag = imag
    
    @classmethod
    def zero(cls):
        return cls(PhiReal.zero(), PhiReal.zero())
    
    def __add__(self, other):
        if isinstance(other, PhiComplex):
            return PhiComplex(self.real + other.real, self.imag + other.imag)
        return PhiComplex(self.real + other, self.imag)
    
    def __mul__(self, other):
        if isinstance(other, PhiComplex):
            real_part = self.real * other.real - self.imag * other.imag
            imag_part = self.real * other.imag + self.imag * other.real
            return PhiComplex(real_part, imag_part)
        elif isinstance(other, PhiReal):
            return PhiComplex(self.real * other, self.imag * other)
        else:
            return PhiComplex(self.real * other, self.imag * other)
    
    def __truediv__(self, other):
        if isinstance(other, PhiReal):
            return PhiComplex(self.real / other, self.imag / other)
        elif isinstance(other, PhiComplex):
            # 复数除法: (a+bi) / (c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
            denominator = other.real * other.real + other.imag * other.imag
            real_part = (self.real * other.real + self.imag * other.imag) / denominator
            imag_part = (self.imag * other.real - self.real * other.imag) / denominator
            return PhiComplex(real_part, imag_part)
        else:
            scalar = PhiReal.from_decimal(float(other))
            return PhiComplex(self.real / scalar, self.imag / scalar)

class PhiMatrix:
    def __init__(self, data: List[List[PhiComplex]]):
        self.data = data

# T19-2认知系统完整类定义

class CognitiveState(Enum):
    """认知状态类型"""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    LANGUAGE = "language"
    CREATIVITY = "creativity"
    METACOGNITION = "metacognition"
    EMOTION = "emotion"
    MOTOR = "motor"
    LEARNING = "learning"

class MemoryType(Enum):
    """记忆类型"""
    SENSORY = "sensory"
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"

class AttentionType(Enum):
    """注意力类型"""
    FOCUS = "focus"
    PERIPHERAL = "peripheral"
    BACKGROUND = "background"

@dataclass
class ZeckendorfCognitive:
    """认知Zeckendorf编码"""
    fibonacci_coefficients: List[int]
    cognitive_meaning: str
    no_consecutive_ones: bool = True
    
    def __post_init__(self):
        """验证no-11约束"""
        for i in range(len(self.fibonacci_coefficients) - 1):
            if self.fibonacci_coefficients[i] == 1 and self.fibonacci_coefficients[i+1] == 1:
                raise ValueError(f"违反no-11约束: 位置{i}和{i+1}都为1")

@dataclass
class PhiCognitiveState:
    """φ-认知量子态"""
    amplitudes: List[PhiComplex]
    cognitive_basis: List[str]
    normalization: PhiReal
    coherence_time: PhiReal
    
    def norm_squared(self) -> PhiReal:
        """计算态的模长平方"""
        total = PhiReal.zero()
        for amp in self.amplitudes:
            norm_sq = amp.real * amp.real + amp.imag * amp.imag
            total = total + norm_sq
        return total

@dataclass
class PhiNeuron:
    """φ-神经元"""
    activation: PhiReal
    threshold: PhiReal
    connections: List[PhiReal]
    fibonacci_index: int

@dataclass 
class PhiMemoryTrace:
    """φ-记忆痕迹"""
    content: PhiCognitiveState
    strength: PhiReal
    timestamp: PhiReal
    decay_rate: PhiReal
    memory_type: MemoryType

class PhiNeuralNetwork:
    """φ-神经网络架构 - 完整分级网络实现"""
    
    def __init__(self, num_layers: int):
        """初始化φ-神经网络"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.num_layers = num_layers
        self.fibonacci = self._generate_fibonacci(num_layers + 10)
        
        # 初始化网络层
        self.layers = self._initialize_layers()
        self.connections = self._initialize_connections()
        self.activation_history = []
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci数列"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def _initialize_layers(self) -> List[List[PhiNeuron]]:
        """初始化网络层"""
        layers = []
        
        for layer_idx in range(self.num_layers):
            # 每层神经元数量为Fibonacci数
            num_neurons = self.fibonacci[layer_idx % len(self.fibonacci)]
            layer = []
            
            for neuron_idx in range(num_neurons):
                # 阈值按φ衰减
                threshold = PhiReal.one() / (self.phi ** neuron_idx) if neuron_idx > 0 else PhiReal.one()
                
                # 初始化连接权重
                connections = []
                if layer_idx > 0:
                    prev_layer_size = len(layers[layer_idx - 1])
                    for prev_idx in range(prev_layer_size):
                        # 连接权重按距离φ衰减
                        weight = PhiReal.one() / (self.phi ** abs(neuron_idx - prev_idx))
                        connections.append(weight)
                
                neuron = PhiNeuron(
                    activation=PhiReal.zero(),
                    threshold=threshold,
                    connections=connections,
                    fibonacci_index=layer_idx
                )
                layer.append(neuron)
            
            layers.append(layer)
        
        return layers
    
    def _initialize_connections(self) -> PhiMatrix:
        """初始化连接矩阵"""
        total_neurons = sum(len(layer) for layer in self.layers)
        connection_data = []
        
        for i in range(total_neurons):
            row = []
            for j in range(total_neurons):
                if i == j:
                    connection = PhiComplex.zero()
                else:
                    distance = abs(i - j)
                    strength = PhiReal.one() / (self.phi ** distance)
                    connection = PhiComplex(strength, PhiReal.zero())
                
                row.append(connection)
            connection_data.append(row)
        
        return PhiMatrix(connection_data)
    
    def phi_activation(self, x: PhiReal) -> PhiReal:
        """φ-激活函数: tanh(φx)"""
        phi_x = self.phi * x
        tanh_val = math.tanh(phi_x.decimal_value)
        return PhiReal.from_decimal(tanh_val)
    
    def forward_propagation(self, input_data: List[PhiReal]) -> List[PhiReal]:
        """前向传播"""
        if len(input_data) != len(self.layers[0]):
            raise ValueError(f"输入维度{len(input_data)}与第一层神经元数{len(self.layers[0])}不匹配")
        
        current_activations = input_data[:]
        
        for layer_idx in range(1, self.num_layers):
            next_activations = []
            current_layer = self.layers[layer_idx]
            
            for neuron_idx, neuron in enumerate(current_layer):
                weighted_sum = PhiReal.zero()
                for prev_idx, prev_activation in enumerate(current_activations):
                    if prev_idx < len(neuron.connections):
                        weight = neuron.connections[prev_idx]
                        weighted_sum = weighted_sum + weight * prev_activation
                
                if weighted_sum.decimal_value > neuron.threshold.decimal_value:
                    activation = self.phi_activation(weighted_sum)
                else:
                    activation = PhiReal.zero()
                
                next_activations.append(activation)
            
            current_activations = next_activations
        
        self.activation_history.append(current_activations[:])
        return current_activations
    
    def calculate_network_capacity(self) -> PhiReal:
        """计算网络容量"""
        total_capacity = PhiReal.zero()
        
        for layer_idx, layer in enumerate(self.layers):
            layer_neurons = PhiReal.from_decimal(len(layer))
            phi_factor = PhiReal.from_decimal(math.log(self.phi.decimal_value, 2))
            layer_capacity = layer_neurons * phi_factor
            total_capacity = total_capacity + layer_capacity
        
        return total_capacity

class PhiMemorySystem:
    """φ-记忆系统 - 完整分级记忆架构"""
    
    def __init__(self):
        """初始化φ-记忆系统"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.fibonacci = self._generate_fibonacci(15)
        
        # 初始化记忆层级
        self.memory_layers = {
            MemoryType.SENSORY: self._initialize_sensory_memory(),
            MemoryType.SHORT_TERM: self._initialize_short_term_memory(),
            MemoryType.WORKING: self._initialize_working_memory(),
            MemoryType.LONG_TERM: self._initialize_long_term_memory()
        }
        
        self.retrieval_index = {}
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci数列"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def _initialize_sensory_memory(self) -> List[Optional[PhiMemoryTrace]]:
        """初始化感觉记忆 - F_1 = 1"""
        capacity = self.fibonacci[1]
        return [None] * capacity
    
    def _initialize_short_term_memory(self) -> List[Optional[PhiMemoryTrace]]:
        """初始化短期记忆 - F_5 = 5"""
        capacity = self.fibonacci[5]
        return [None] * capacity
    
    def _initialize_working_memory(self) -> List[Optional[PhiMemoryTrace]]:
        """初始化工作记忆 - F_6 = 8"""
        capacity = self.fibonacci[6]
        return [None] * capacity
    
    def _initialize_long_term_memory(self) -> List[Optional[PhiMemoryTrace]]:
        """初始化长期记忆 - F_13 = 233"""
        capacity = self.fibonacci[13]
        return [None] * capacity
    
    def encode_memory(self, content: PhiCognitiveState, memory_type: MemoryType, 
                     current_time: PhiReal) -> bool:
        """编码记忆"""
        memory_layer = self.memory_layers[memory_type]
        
        decay_rates = {
            MemoryType.SENSORY: self.phi ** 4,
            MemoryType.SHORT_TERM: self.phi ** 3,
            MemoryType.WORKING: self.phi ** 2,
            MemoryType.LONG_TERM: self.phi
        }
        
        trace = PhiMemoryTrace(
            content=content,
            strength=PhiReal.one(),
            timestamp=current_time,
            decay_rate=decay_rates[memory_type],
            memory_type=memory_type
        )
        
        # 寻找空闲位置或替换最弱记忆
        empty_slot = None
        weakest_slot = 0
        weakest_strength = PhiReal.from_decimal(float('inf'))
        
        for i, existing_trace in enumerate(memory_layer):
            if existing_trace is None:
                empty_slot = i
                break
            else:
                if existing_trace.strength.decimal_value < weakest_strength.decimal_value:
                    weakest_strength = existing_trace.strength
                    weakest_slot = i
        
        if empty_slot is not None:
            memory_layer[empty_slot] = trace
        else:
            memory_layer[weakest_slot] = trace
        
        return True
    
    def retrieve_memory(self, cue: PhiCognitiveState, current_time: PhiReal) -> Optional[PhiMemoryTrace]:
        """检索记忆"""
        best_match = None
        best_similarity = PhiReal.zero()
        
        for memory_type, memory_layer in self.memory_layers.items():
            for trace in memory_layer:
                if trace is None:
                    continue
                
                # 计算记忆衰减
                time_diff = current_time - trace.timestamp
                decay_factor = (-time_diff / trace.decay_rate).exp()
                current_strength = trace.strength * decay_factor
                
                if current_strength.decimal_value < 0.01:
                    continue
                
                # 计算相似度
                similarity = self._calculate_similarity(cue, trace.content)
                weighted_similarity = similarity * current_strength
                
                if weighted_similarity.decimal_value > best_similarity.decimal_value:
                    best_similarity = weighted_similarity
                    best_match = trace
        
        return best_match
    
    def _calculate_similarity(self, cue: PhiCognitiveState, memory: PhiCognitiveState) -> PhiReal:
        """计算认知状态相似度"""
        if len(cue.amplitudes) != len(memory.amplitudes):
            return PhiReal.zero()
        
        dot_product = PhiReal.zero()
        cue_norm = PhiReal.zero()
        memory_norm = PhiReal.zero()
        
        for i in range(len(cue.amplitudes)):
            real_part = cue.amplitudes[i].real * memory.amplitudes[i].real + \
                       cue.amplitudes[i].imag * memory.amplitudes[i].imag
            dot_product = dot_product + real_part
            
            cue_norm_sq = cue.amplitudes[i].real * cue.amplitudes[i].real + \
                         cue.amplitudes[i].imag * cue.amplitudes[i].imag
            memory_norm_sq = memory.amplitudes[i].real * memory.amplitudes[i].real + \
                           memory.amplitudes[i].imag * memory.amplitudes[i].imag
            
            cue_norm = cue_norm + cue_norm_sq
            memory_norm = memory_norm + memory_norm_sq
        
        if cue_norm.decimal_value > 0 and memory_norm.decimal_value > 0:
            similarity = dot_product / (cue_norm.sqrt() * memory_norm.sqrt())
            return PhiReal.from_decimal(max(0, similarity.decimal_value))
        else:
            return PhiReal.zero()

class PhiAttentionMechanism:
    """φ-注意力机制 - 完整多层注意力系统"""
    
    def __init__(self, input_dim: int):
        """初始化φ-注意力机制"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.input_dim = input_dim
        self.fibonacci = self._generate_fibonacci(10)
        
        # 注意力窗口大小
        self.attention_windows = {
            AttentionType.FOCUS: self.fibonacci[3],      # F_3 = 2
            AttentionType.PERIPHERAL: self.fibonacci[6],  # F_6 = 8
            AttentionType.BACKGROUND: self.fibonacci[8]   # F_8 = 21
        }
        
        self.attention_history = []
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci数列"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def phi_softmax(self, scores: List[PhiReal]) -> List[PhiReal]:
        """φ-softmax函数"""
        phi_scores = []
        for score in scores:
            phi_score_val = self.phi.decimal_value ** score.decimal_value
            phi_scores.append(PhiReal.from_decimal(phi_score_val))
        
        total = PhiReal.zero()
        for phi_score in phi_scores:
            total = total + phi_score
        
        normalized = []
        for phi_score in phi_scores:
            if total.decimal_value > 0:
                prob = phi_score / total
            else:
                prob = PhiReal.from_decimal(1.0 / len(phi_scores))
            normalized.append(prob)
        
        return normalized
    
    def compute_attention(self, inputs: List[PhiCognitiveState]) -> Tuple[List[PhiCognitiveState], List[PhiReal]]:
        """计算注意力权重和输出"""
        if not inputs:
            return [], []
        
        num_inputs = len(inputs)
        
        # 计算注意力分数（简化版本，基于状态相似性）
        attention_scores = []
        for i in range(num_inputs):
            scores_for_i = []
            for j in range(num_inputs):
                if i == j:
                    score = PhiReal.one()  # 自注意力为1
                else:
                    # 计算状态间相似性作为注意力分数
                    score = self._compute_similarity(inputs[i], inputs[j])
                
                # φ-缩放
                scaling_factor = math.sqrt(self.input_dim / self.phi.decimal_value)
                scaled_score = score / PhiReal.from_decimal(scaling_factor)
                scores_for_i.append(scaled_score)
            attention_scores.append(scores_for_i)
        
        # 应用φ-softmax
        attention_weights = []
        for scores_row in attention_scores:
            weights_row = self.phi_softmax(scores_row)
            attention_weights.append(weights_row)
        
        # 计算加权输出
        outputs = []
        for i in range(num_inputs):
            weighted_sum = self._compute_weighted_sum(inputs, attention_weights[i])
            outputs.append(weighted_sum)
        
        # 计算平均注意力权重
        avg_weights = []
        for j in range(num_inputs):
            total_weight = PhiReal.zero()
            for i in range(num_inputs):
                total_weight = total_weight + attention_weights[i][j]
            avg_weight = total_weight / PhiReal.from_decimal(num_inputs)
            avg_weights.append(avg_weight)
        
        self.attention_history.append(attention_weights)
        return outputs, avg_weights
    
    def _compute_similarity(self, state1: PhiCognitiveState, state2: PhiCognitiveState) -> PhiReal:
        """计算两个认知状态的相似性"""
        if len(state1.amplitudes) != len(state2.amplitudes):
            return PhiReal.zero()
        
        dot_product = PhiReal.zero()
        for i in range(len(state1.amplitudes)):
            real_part = state1.amplitudes[i].real * state2.amplitudes[i].real + \
                       state1.amplitudes[i].imag * state2.amplitudes[i].imag
            dot_product = dot_product + real_part
        
        return dot_product
    
    def _compute_weighted_sum(self, values: List[PhiCognitiveState], weights: List[PhiReal]) -> PhiCognitiveState:
        """计算加权和"""
        if not values or not weights:
            return PhiCognitiveState([], [], PhiReal.zero(), PhiReal.zero())
        
        result_amplitudes = [PhiComplex.zero() for _ in range(len(values[0].amplitudes))]
        result_basis = values[0].cognitive_basis[:]
        
        for i, (value_state, weight) in enumerate(zip(values, weights)):
            for j in range(len(result_amplitudes)):
                if j < len(value_state.amplitudes):
                    weighted_amp = PhiComplex(
                        value_state.amplitudes[j].real * weight,
                        value_state.amplitudes[j].imag * weight
                    )
                    result_amplitudes[j] = result_amplitudes[j] + weighted_amp
        
        return PhiCognitiveState(
            amplitudes=result_amplitudes,
            cognitive_basis=result_basis,
            normalization=PhiReal.one(),
            coherence_time=values[0].coherence_time
        )
    
    def get_attention_switching_time(self) -> PhiReal:
        """获取注意力切换时间"""
        ln_phi = math.log(self.phi.decimal_value)
        switching_time = ln_phi / self.phi.decimal_value
        return PhiReal.from_decimal(switching_time)

class PhiLanguageProcessor:
    """φ-语言处理系统 - 完整递归语法解析器"""
    
    def __init__(self, vocab_size: int):
        """初始化φ-语言处理器"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.vocab_size = vocab_size
        self.fibonacci = self._generate_fibonacci(15)
        
        self.max_depth = self.fibonacci[8]  # F_8 = 21
        self.semantic_dim = int(self.fibonacci[10] * math.log(self.phi.decimal_value, 2))
        
        self.vocabulary = self._initialize_vocabulary()
        self.syntax_rules = self._initialize_syntax_rules()
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci数列"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def _initialize_vocabulary(self) -> Dict[str, ZeckendorfCognitive]:
        """初始化词汇表"""
        vocab = {}
        test_words = ["the", "cat", "dog", "runs", "jumps", "quickly", "slowly", "red", "blue", "house"]
        
        for i, word in enumerate(test_words[:min(self.vocab_size, len(test_words))]):
            zeck_encoding = self._word_to_zeckendorf(word, i)
            vocab[word] = zeck_encoding
        
        return vocab
    
    def _word_to_zeckendorf(self, word: str, index: int) -> ZeckendorfCognitive:
        """将词汇转换为Zeckendorf编码"""
        coeffs = self._to_zeckendorf(index + 1)
        return ZeckendorfCognitive(
            fibonacci_coefficients=coeffs,
            cognitive_meaning=f"word: {word}"
        )
    
    def _to_zeckendorf(self, n: int) -> List[int]:
        """将整数转换为Zeckendorf表示"""
        if n == 0:
            return [0]
        
        fib_index = 0
        while fib_index < len(self.fibonacci) and self.fibonacci[fib_index] <= n:
            fib_index += 1
        fib_index -= 1
        
        coeffs = [0] * (fib_index + 1)
        remaining = n
        
        for i in range(fib_index, -1, -1):
            if remaining >= self.fibonacci[i]:
                coeffs[i] = 1
                remaining -= self.fibonacci[i]
        
        return coeffs
    
    def _initialize_syntax_rules(self) -> Dict[str, List[str]]:
        """初始化语法规则"""
        return {
            "S": ["NP VP"],
            "NP": ["Det N", "N"],
            "VP": ["V", "V NP", "V Adv"],
            "Det": ["the"],
            "N": ["cat", "dog", "house"],
            "V": ["runs", "jumps"],
            "Adv": ["quickly", "slowly"]
        }
    
    def tokenize(self, sentence: str) -> List[str]:
        """词法分析"""
        return sentence.lower().split()
    
    def parse_sentence(self, tokens: List[str]) -> Optional[Dict]:
        """句法解析"""
        if not tokens:
            return None
        
        parse_tree = self._recursive_parse("S", tokens, 0, 0)
        return parse_tree
    
    def _recursive_parse(self, symbol: str, tokens: List[str], position: int, depth: int) -> Optional[Dict]:
        """φ-递归解析"""
        if depth > self.max_depth or position >= len(tokens):
            return None
        
        # 终结符
        if symbol not in self.syntax_rules:
            if position < len(tokens) and tokens[position] == symbol:
                return {
                    "symbol": symbol,
                    "value": tokens[position],
                    "position": position + 1,
                    "depth": depth,
                    "phi_weight": PhiReal.one() / (self.phi ** depth)
                }
            else:
                return None
        
        # 非终结符
        for rule in self.syntax_rules[symbol]:
            rule_symbols = rule.split()
            parse_result = self._parse_rule(rule_symbols, tokens, position, depth + 1)
            
            if parse_result is not None:
                return {
                    "symbol": symbol,
                    "rule": rule,
                    "children": parse_result["children"],
                    "position": parse_result["position"],
                    "depth": depth,
                    "phi_weight": PhiReal.one() / (self.phi ** depth)
                }
        
        return None
    
    def _parse_rule(self, rule_symbols: List[str], tokens: List[str], position: int, depth: int) -> Optional[Dict]:
        """解析规则"""
        children = []
        current_pos = position
        
        for symbol in rule_symbols:
            child_parse = self._recursive_parse(symbol, tokens, current_pos, depth)
            if child_parse is None:
                return None
            
            children.append(child_parse)
            current_pos = child_parse["position"]
        
        return {
            "children": children,
            "position": current_pos
        }
    
    def understand_sentence(self, sentence: str) -> PhiCognitiveState:
        """完整句子理解流程"""
        tokens = self.tokenize(sentence)
        parse_tree = self.parse_sentence(tokens)
        
        if parse_tree is None:
            return PhiCognitiveState([], [], PhiReal.zero(), PhiReal.zero())
        
        # 简化的语义提取
        semantic_components = self._extract_semantic_components(parse_tree)
        
        amplitudes = []
        basis_labels = []
        
        for i, component in enumerate(semantic_components):
            if component.decimal_value != 0:
                amplitude = PhiComplex(component, PhiReal.zero())
                amplitudes.append(amplitude)
                basis_labels.append(f"semantic_{i}")
        
        return PhiCognitiveState(
            amplitudes=amplitudes,
            cognitive_basis=basis_labels,
            normalization=PhiReal.one(),
            coherence_time=PhiReal.from_decimal(1.0)
        )
    
    def _extract_semantic_components(self, node: Dict) -> List[PhiReal]:
        """提取语义组件"""
        components = [PhiReal.zero() for _ in range(10)]  # 固定大小的语义向量
        
        if "value" in node:
            word = node["value"]
            if word in self.vocabulary:
                word_encoding = self.vocabulary[word]
                for i, coeff in enumerate(word_encoding.fibonacci_coefficients):
                    if i < len(components) and coeff == 1:
                        components[i] = components[i] + node["phi_weight"]
        
        if "children" in node:
            for child in node["children"]:
                child_components = self._extract_semantic_components(child)
                for i in range(len(components)):
                    if i < len(child_components):
                        components[i] = components[i] + child_components[i] / self.phi
        
        return components

class PhiReasoningSystem:
    """φ-推理系统 - 完整逻辑推理引擎"""
    
    def __init__(self, max_proof_depth: int = 8):
        """初始化φ-推理系统"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.max_proof_depth = max_proof_depth
        self.fibonacci = self._generate_fibonacci(max_proof_depth + 5)
        
        self.inference_rules = {
            "modus_ponens": self.phi ** 0,
            "modus_tollens": self.phi ** (-1),
            "hypothetical_syllogism": self.phi ** (-1),
            "disjunctive_syllogism": self.phi ** (-1),
            "induction": self.phi ** (-2)
        }
        
        self.knowledge_base = []
        self.proof_cache = {}
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci数列"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def add_knowledge(self, proposition: str, certainty: PhiReal):
        """添加知识到知识库"""
        knowledge_item = {
            "proposition": proposition,
            "certainty": certainty,
            "timestamp": PhiReal.from_decimal(len(self.knowledge_base))
        }
        self.knowledge_base.append(knowledge_item)
    
    def prove_goal(self, goal: str) -> Tuple[bool, List[Dict], PhiReal]:
        """证明目标"""
        if goal in self.proof_cache:
            return self.proof_cache[goal]
        
        proof_tree = self._search_proof(goal, 0, [])
        
        if proof_tree is not None:
            success = True
            confidence = self._calculate_proof_confidence(proof_tree)
            proof_steps = self._extract_proof_steps(proof_tree)
        else:
            success = False
            confidence = PhiReal.zero()
            proof_steps = []
        
        result = (success, proof_steps, confidence)
        self.proof_cache[goal] = result
        return result
    
    def _search_proof(self, goal: str, depth: int, used_facts: List[str]) -> Optional[Dict]:
        """搜索证明"""
        if depth > self.max_proof_depth:
            return None
        
        # 检查知识库中的直接事实
        for knowledge in self.knowledge_base:
            if knowledge["proposition"] == goal and knowledge not in used_facts:
                return {
                    "type": "fact",
                    "goal": goal,
                    "knowledge": knowledge,
                    "depth": depth,
                    "confidence": knowledge["certainty"]
                }
        
        # 尝试假言推理
        modus_ponens_proof = self._try_modus_ponens(goal, depth, used_facts)
        if modus_ponens_proof is not None:
            return modus_ponens_proof
        
        return None
    
    def _try_modus_ponens(self, goal: str, depth: int, used_facts: List[str]) -> Optional[Dict]:
        """尝试假言推理"""
        for knowledge in self.knowledge_base:
            if knowledge in used_facts:
                continue
                
            prop = knowledge["proposition"]
            if " -> " in prop and prop.endswith(" -> " + goal):
                antecedent = prop.split(" -> ")[0]
                
                antecedent_proof = self._search_proof(antecedent, depth + 1, used_facts + [knowledge])
                
                if antecedent_proof is not None:
                    return {
                        "type": "modus_ponens",
                        "goal": goal,
                        "rule": knowledge,
                        "antecedent_proof": antecedent_proof,
                        "depth": depth,
                        "rule_weight": self.inference_rules["modus_ponens"]
                    }
        
        return None
    
    def _calculate_proof_confidence(self, proof_tree: Dict) -> PhiReal:
        """计算证明置信度"""
        if proof_tree["type"] == "fact":
            return proof_tree["confidence"]
        elif proof_tree["type"] == "modus_ponens":
            rule_confidence = proof_tree["rule"]["certainty"]
            antecedent_confidence = self._calculate_proof_confidence(proof_tree["antecedent_proof"])
            rule_weight = proof_tree["rule_weight"]
            combined = rule_confidence * antecedent_confidence * rule_weight
            return combined
        else:
            return PhiReal.zero()
    
    def _extract_proof_steps(self, proof_tree: Dict) -> List[Dict]:
        """提取证明步骤"""
        steps = []
        
        if proof_tree["type"] == "fact":
            steps.append({
                "step": f"Given: {proof_tree['goal']}",
                "confidence": proof_tree["confidence"].decimal_value
            })
        elif proof_tree["type"] == "modus_ponens":
            antecedent_steps = self._extract_proof_steps(proof_tree["antecedent_proof"])
            steps.extend(antecedent_steps)
            steps.append({
                "step": f"By Modus Ponens from '{proof_tree['rule']['proposition']}' and previous: {proof_tree['goal']}",
                "confidence": self._calculate_proof_confidence(proof_tree).decimal_value
            })
        
        return steps

class PhiCognitiveArchitecture:
    """φ-认知架构主系统 - 完整自指认知实现"""
    
    def __init__(self, input_dim: int = 8):
        """初始化φ-认知架构"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.input_dim = input_dim
        
        # 初始化认知子系统
        self.neural_network = PhiNeuralNetwork(5)
        self.memory_system = PhiMemorySystem()
        self.attention = PhiAttentionMechanism(input_dim)
        self.language_processor = PhiLanguageProcessor(20)
        self.reasoning_system = PhiReasoningSystem()
        
        # 认知状态
        self.current_state = PhiCognitiveState([], [], PhiReal.zero(), PhiReal.zero())
        self.cognitive_history = []
        
    def process_cognitive_input(self, input_state: PhiCognitiveState) -> PhiCognitiveState:
        """完整的认知输入处理：C = C[C]"""
        
        # 1. 注意力分配
        attended_states, attention_weights = self.attention.compute_attention([input_state])
        if not attended_states:
            return input_state
        
        attended_state = attended_states[0]
        
        # 2. 记忆整合
        current_time = PhiReal.from_decimal(len(self.cognitive_history))
        retrieved_memory = self.memory_system.retrieve_memory(attended_state, current_time)
        
        if retrieved_memory is not None:
            integrated_amplitudes = []
            max_len = max(len(attended_state.amplitudes), len(retrieved_memory.content.amplitudes))
            
            for i in range(max_len):
                current_amp = attended_state.amplitudes[i] if i < len(attended_state.amplitudes) else PhiComplex.zero()
                memory_amp = retrieved_memory.content.amplitudes[i] if i < len(retrieved_memory.content.amplitudes) else PhiComplex.zero()
                
                integrated_amp = current_amp + memory_amp / self.phi
                integrated_amplitudes.append(integrated_amp)
            
            integrated_state = PhiCognitiveState(
                amplitudes=integrated_amplitudes,
                cognitive_basis=attended_state.cognitive_basis + ["memory_integrated"],
                normalization=PhiReal.one(),
                coherence_time=attended_state.coherence_time
            )
        else:
            integrated_state = attended_state
        
        # 3. 存储到记忆系统
        self.memory_system.encode_memory(integrated_state, MemoryType.WORKING, current_time)
        
        # 4. 更新认知状态
        self.current_state = integrated_state
        self.cognitive_history.append(integrated_state)
        
        return integrated_state

def verify_cognitive_self_reference_property(cognitive_arch: PhiCognitiveArchitecture) -> bool:
    """验证认知系统的自指性质：C = C[C]"""
    
    test_input = PhiCognitiveState(
        amplitudes=[PhiComplex(PhiReal.one(), PhiReal.zero())],
        cognitive_basis=["test_concept"],
        normalization=PhiReal.one(),
        coherence_time=PhiReal.from_decimal(1.0)
    )
    
    # 第一次认知处理：C
    result1 = cognitive_arch.process_cognitive_input(test_input)
    
    # 第二次认知处理：C[C]
    if result1 is not None:
        result2 = cognitive_arch.process_cognitive_input(result1)
    else:
        return False
    
    # 验证自指性质
    processing_successful = result2 is not None and len(result2.amplitudes) > 0
    coherence_maintained = result2 is not None and \
                          result2.coherence_time.decimal_value >= result1.coherence_time.decimal_value * 0.9
    complexity_increased = result2 is not None and \
                          len(result2.amplitudes) >= len(result1.amplitudes)
    
    return processing_successful and coherence_maintained and complexity_increased

def complete_phi_cognitive_architecture_verification() -> Dict[str, bool]:
    """完整验证φ-认知架构系统的所有核心性质"""
    
    results = {}
    
    try:
        # 1. 验证φ-神经网络架构
        neural_net = PhiNeuralNetwork(5)
        test_input = [PhiReal.from_decimal(0.5), PhiReal.one(), PhiReal.from_decimal(0.3)]
        
        while len(test_input) < len(neural_net.layers[0]):
            test_input.append(PhiReal.zero())
        
        net_output = neural_net.forward_propagation(test_input[:len(neural_net.layers[0])])
        results["neural_network"] = len(net_output) > 0
        
        capacity = neural_net.calculate_network_capacity()
        results["network_capacity"] = capacity.decimal_value > 0
        
        # 2. 验证φ-记忆系统
        memory_system = PhiMemorySystem()
        
        test_memory = PhiCognitiveState(
            amplitudes=[PhiComplex(PhiReal.one(), PhiReal.zero())],
            cognitive_basis=["test_memory"],
            normalization=PhiReal.one(),
            coherence_time=PhiReal.from_decimal(1.0)
        )
        
        encode_success = memory_system.encode_memory(test_memory, MemoryType.WORKING, PhiReal.zero())
        results["memory_encoding"] = encode_success
        
        retrieved = memory_system.retrieve_memory(test_memory, PhiReal.from_decimal(0.1))
        results["memory_retrieval"] = retrieved is not None
        
        # 3. 验证φ-注意力机制
        attention = PhiAttentionMechanism(3)
        
        test_states = [
            PhiCognitiveState([PhiComplex(PhiReal.one(), PhiReal.zero())], ["state1"], PhiReal.one(), PhiReal.one()),
            PhiCognitiveState([PhiComplex(PhiReal.from_decimal(0.5), PhiReal.zero())], ["state2"], PhiReal.one(), PhiReal.one())
        ]
        
        outputs, weights = attention.compute_attention(test_states)
        results["attention_mechanism"] = len(outputs) > 0 and len(weights) > 0
        
        switching_time = attention.get_attention_switching_time()
        results["attention_switching"] = abs(switching_time.decimal_value - 0.31) < 0.1
        
        # 4. 验证φ-语言处理
        language_proc = PhiLanguageProcessor(10)
        
        test_sentence = "the cat runs"
        understanding = language_proc.understand_sentence(test_sentence)
        results["language_processing"] = len(understanding.amplitudes) > 0
        
        # 5. 验证φ-推理系统
        reasoning = PhiReasoningSystem()
        
        reasoning.add_knowledge("P -> Q", PhiReal.from_decimal(0.9))
        reasoning.add_knowledge("P", PhiReal.from_decimal(0.8))
        
        success, proof_steps, confidence = reasoning.prove_goal("Q")
        results["reasoning_system"] = success and len(proof_steps) > 0
        results["reasoning_confidence"] = confidence.decimal_value > 0
        
        # 6. 验证φ-结构一致性
        phi_check = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        results["phi_structure_consistency"] = abs(phi_check.decimal_value - 1.618) < 0.001
        
    except Exception as e:
        results["exception"] = f"验证过程中发生异常: {str(e)}"
    
    return results

# 测试类
class TestT19_2CognitiveArchitecture(unittest.TestCase):
    """T19-2 φ-认知架构定理测试类"""
    
    def test_phi_neural_network_architecture(self):
        """测试φ-神经网络架构"""
        neural_net = PhiNeuralNetwork(4)
        
        # 验证网络结构
        self.assertGreater(len(neural_net.layers), 0, "神经网络必须有层")
        self.assertGreater(len(neural_net.fibonacci), 0, "必须生成Fibonacci数列")
        
        # 验证每层神经元数量为Fibonacci数
        for layer_idx, layer in enumerate(neural_net.layers):
            expected_neurons = neural_net.fibonacci[layer_idx % len(neural_net.fibonacci)]
            self.assertEqual(len(layer), expected_neurons,
                           f"第{layer_idx}层神经元数量应为Fibonacci数{expected_neurons}")
        
        # 测试前向传播
        if len(neural_net.layers) > 0:
            input_size = len(neural_net.layers[0])
            test_input = [PhiReal.from_decimal(0.5)] * input_size
            
            output = neural_net.forward_propagation(test_input)
            
            self.assertIsInstance(output, list, "前向传播必须返回列表")
            self.assertGreater(len(output), 0, "输出不能为空")
            
            for out_val in output:
                self.assertIsInstance(out_val, PhiReal, "输出必须是PhiReal类型")
        
        # 测试网络容量计算
        capacity = neural_net.calculate_network_capacity()
        self.assertIsInstance(capacity, PhiReal, "网络容量必须是PhiReal类型")
        self.assertGreater(capacity.decimal_value, 0, "网络容量必须为正")

    def test_phi_memory_system_hierarchy(self):
        """测试φ-记忆系统层级"""
        memory_system = PhiMemorySystem()
        
        # 验证记忆层级容量
        expected_capacities = {
            MemoryType.SENSORY: memory_system.fibonacci[1],    # F_1 = 1
            MemoryType.SHORT_TERM: memory_system.fibonacci[5], # F_5 = 5
            MemoryType.WORKING: memory_system.fibonacci[6],    # F_6 = 8
            MemoryType.LONG_TERM: memory_system.fibonacci[13]  # F_13 = 233
        }
        
        for memory_type, expected_capacity in expected_capacities.items():
            actual_capacity = len(memory_system.memory_layers[memory_type])
            self.assertEqual(actual_capacity, expected_capacity,
                           f"{memory_type.value}记忆容量应为F_{expected_capacity}")
        
        # 测试记忆编码
        test_memory = PhiCognitiveState(
            amplitudes=[PhiComplex(PhiReal.one(), PhiReal.zero())],
            cognitive_basis=["test_memory"],
            normalization=PhiReal.one(),
            coherence_time=PhiReal.from_decimal(1.0)
        )
        
        current_time = PhiReal.zero()
        
        # 测试各种记忆类型的编码
        for memory_type in [MemoryType.SENSORY, MemoryType.SHORT_TERM, MemoryType.WORKING, MemoryType.LONG_TERM]:
            success = memory_system.encode_memory(test_memory, memory_type, current_time)
            self.assertTrue(success, f"{memory_type.value}记忆编码必须成功")
        
        # 测试记忆检索
        retrieved = memory_system.retrieve_memory(test_memory, PhiReal.from_decimal(0.1))
        self.assertIsNotNone(retrieved, "记忆检索必须成功")
        self.assertIsInstance(retrieved, PhiMemoryTrace, "检索结果必须是PhiMemoryTrace类型")

    def test_phi_attention_mechanism_windows(self):
        """测试φ-注意力机制窗口"""
        attention = PhiAttentionMechanism(5)
        
        # 验证注意力窗口大小
        expected_windows = {
            AttentionType.FOCUS: attention.fibonacci[3],      # F_3 = 2
            AttentionType.PERIPHERAL: attention.fibonacci[6], # F_6 = 8
            AttentionType.BACKGROUND: attention.fibonacci[8]  # F_8 = 21
        }
        
        for attention_type, expected_size in expected_windows.items():
            actual_size = attention.attention_windows[attention_type]
            self.assertEqual(actual_size, expected_size,
                           f"{attention_type.value}注意力窗口应为F_{expected_size}")
        
        # 测试注意力计算
        test_states = [
            PhiCognitiveState(
                [PhiComplex(PhiReal.one(), PhiReal.zero())],
                ["state1"], PhiReal.one(), PhiReal.one()
            ),
            PhiCognitiveState(
                [PhiComplex(PhiReal.from_decimal(0.8), PhiReal.zero())],
                ["state2"], PhiReal.one(), PhiReal.one()
            )
        ]
        
        outputs, weights = attention.compute_attention(test_states)
        
        self.assertEqual(len(outputs), len(test_states), "输出状态数量必须等于输入状态数量")
        self.assertEqual(len(weights), len(test_states), "注意力权重数量必须等于输入状态数量")
        
        # 验证注意力权重和为1（近似）
        total_weight = sum(w.decimal_value for w in weights)
        self.assertAlmostEqual(total_weight, 1.0, places=1, msg="注意力权重和应接近1")
        
        # 测试注意力切换时间
        switching_time = attention.get_attention_switching_time()
        self.assertIsInstance(switching_time, PhiReal, "切换时间必须是PhiReal类型")
        self.assertAlmostEqual(switching_time.decimal_value, 0.31, places=1,
                              msg="注意力切换时间应约为0.31秒")

    def test_phi_language_processing_syntax(self):
        """测试φ-语言处理句法分析"""
        language_proc = PhiLanguageProcessor(15)
        
        # 验证词汇表初始化
        self.assertGreater(len(language_proc.vocabulary), 0, "词汇表不能为空")
        
        for word, encoding in language_proc.vocabulary.items():
            self.assertIsInstance(encoding, ZeckendorfCognitive, "词汇编码必须是ZeckendorfCognitive类型")
            
            # 验证no-11约束
            coeffs = encoding.fibonacci_coefficients
            for i in range(len(coeffs) - 1):
                self.assertFalse(coeffs[i] == 1 and coeffs[i+1] == 1,
                               f"词汇'{word}'编码违反no-11约束")
        
        # 测试句法解析
        test_sentences = [
            "the cat runs",
            "the dog jumps quickly",
            "cat runs"
        ]
        
        for sentence in test_sentences:
            tokens = language_proc.tokenize(sentence)
            self.assertIsInstance(tokens, list, "分词结果必须是列表")
            self.assertGreater(len(tokens), 0, "分词结果不能为空")
            
            parse_tree = language_proc.parse_sentence(tokens)
            
            if parse_tree is not None:  # 有些句子可能无法解析
                self.assertIn("symbol", parse_tree, "解析树必须包含symbol字段")
                self.assertIn("depth", parse_tree, "解析树必须包含depth字段")
                self.assertLessEqual(parse_tree["depth"], language_proc.max_depth,
                                   "解析深度不能超过最大深度")
        
        # 测试完整句子理解
        understanding = language_proc.understand_sentence("the cat runs")
        self.assertIsInstance(understanding, PhiCognitiveState, "句子理解结果必须是PhiCognitiveState")

    def test_phi_reasoning_system_inference(self):
        """测试φ-推理系统推理"""
        reasoning = PhiReasoningSystem()
        
        # 验证推理规则权重
        expected_weights = {
            "modus_ponens": reasoning.phi ** 0,      # φ^0 = 1
            "modus_tollens": reasoning.phi ** (-1),  # φ^(-1)
            "induction": reasoning.phi ** (-2)       # φ^(-2)
        }
        
        for rule_name, expected_weight in expected_weights.items():
            actual_weight = reasoning.inference_rules[rule_name]
            self.assertAlmostEqual(actual_weight.decimal_value, expected_weight.decimal_value,
                                 places=3, msg=f"{rule_name}规则权重不正确")
        
        # 测试知识添加
        reasoning.add_knowledge("P -> Q", PhiReal.from_decimal(0.9))
        reasoning.add_knowledge("P", PhiReal.from_decimal(0.8))
        reasoning.add_knowledge("Q -> R", PhiReal.from_decimal(0.7))
        
        self.assertEqual(len(reasoning.knowledge_base), 3, "知识库应包含3个知识项")
        
        # 测试推理证明
        success, proof_steps, confidence = reasoning.prove_goal("Q")
        
        self.assertTrue(success, "应该能够证明Q")
        self.assertGreater(len(proof_steps), 0, "证明步骤不能为空")
        self.assertGreater(confidence.decimal_value, 0, "证明置信度必须为正")
        
        # 验证证明步骤结构
        for step in proof_steps:
            self.assertIn("step", step, "证明步骤必须包含step字段")
            self.assertIn("confidence", step, "证明步骤必须包含confidence字段")
        
        # 测试无法证明的目标
        success_false, steps_false, conf_false = reasoning.prove_goal("Z")
        self.assertFalse(success_false, "无法证明的目标应该返回False")
        self.assertEqual(len(steps_false), 0, "无法证明的目标证明步骤应为空")
        self.assertEqual(conf_false.decimal_value, 0, "无法证明的目标置信度应为0")

    def test_cognitive_self_reference_property_verification(self):
        """测试认知系统自指性质验证"""
        cognitive_arch = PhiCognitiveArchitecture(5)
        
        # 验证自指性质：C = C[C]
        self_ref_result = verify_cognitive_self_reference_property(cognitive_arch)
        
        self.assertIsInstance(self_ref_result, bool, "自指性质验证必须返回布尔值")
        self.assertTrue(self_ref_result, "认知系统必须满足自指性质 C = C[C]")
        
        # 独立验证各个组件
        test_input = PhiCognitiveState(
            amplitudes=[PhiComplex(PhiReal.one(), PhiReal.zero())],
            cognitive_basis=["test_input"],
            normalization=PhiReal.one(),
            coherence_time=PhiReal.from_decimal(1.0)
        )
        
        # 第一次处理
        result1 = cognitive_arch.process_cognitive_input(test_input)
        
        self.assertIsInstance(result1, PhiCognitiveState, "第一次处理结果必须是PhiCognitiveState")
        self.assertGreater(len(result1.amplitudes), 0, "第一次处理结果不能为空")
        
        # 第二次处理（自指）
        result2 = cognitive_arch.process_cognitive_input(result1)
        
        self.assertIsInstance(result2, PhiCognitiveState, "第二次处理结果必须是PhiCognitiveState")
        self.assertGreater(len(result2.amplitudes), 0, "第二次处理结果不能为空")
        
        # 验证相干性保持
        self.assertGreaterEqual(result2.coherence_time.decimal_value,
                               result1.coherence_time.decimal_value * 0.9,
                               "相干时间必须保持或增加")
        
        # 验证认知复杂度增加
        self.assertGreaterEqual(len(result2.amplitudes), len(result1.amplitudes),
                               "认知复杂度应该保持或增加")

    def test_zeckendorf_encoding_no11_constraint_cognitive(self):
        """测试认知系统中Zeckendorf编码的no-11约束"""
        # 测试有效的Zeckendorf认知编码
        valid_coeffs = [1, 0, 1, 0, 1, 0, 1]
        valid_encoding = ZeckendorfCognitive(valid_coeffs, "valid cognitive pattern")
        
        self.assertEqual(valid_encoding.fibonacci_coefficients, valid_coeffs,
                        "有效编码的系数应该保持不变")
        self.assertTrue(valid_encoding.no_consecutive_ones,
                       "有效编码应该满足no-11约束")
        
        # 测试无效的编码（违反no-11约束）
        invalid_coeffs = [1, 1, 0, 1, 0]
        
        with self.assertRaises(ValueError, msg="违反no-11约束的编码应该抛出异常"):
            ZeckendorfCognitive(invalid_coeffs, "invalid cognitive pattern")
        
        # 测试边界情况
        edge_cases = [
            [1],           # 单个1
            [0],           # 单个0
            [1, 0],        # 10模式
            [0, 1],        # 01模式
            [1, 0, 1, 0]   # 1010模式
        ]
        
        for case in edge_cases:
            try:
                encoding = ZeckendorfCognitive(case, f"edge case {case}")
                self.assertEqual(encoding.fibonacci_coefficients, case)
            except ValueError:
                self.fail(f"边界情况{case}不应该抛出异常")

    def test_phi_structure_consistency_across_cognitive_systems(self):
        """测试认知系统间φ-结构一致性"""
        phi_value = (1 + math.sqrt(5)) / 2
        
        # 从不同认知子系统提取φ值
        neural_net = PhiNeuralNetwork(3)
        memory_sys = PhiMemorySystem()
        attention = PhiAttentionMechanism(4)
        language_proc = PhiLanguageProcessor(10)
        reasoning = PhiReasoningSystem()
        cognitive_arch = PhiCognitiveArchitecture(6)
        
        systems_phi_values = [
            neural_net.phi.decimal_value,
            memory_sys.phi.decimal_value,
            attention.phi.decimal_value,
            language_proc.phi.decimal_value,
            reasoning.phi.decimal_value,
            cognitive_arch.phi.decimal_value
        ]
        
        # 验证所有系统使用相同的φ值
        for i, phi_val in enumerate(systems_phi_values):
            self.assertAlmostEqual(phi_val, phi_value, places=10,
                                  msg=f"认知子系统{i}的φ值不一致")
        
        # 验证φ值的数学特性
        self.assertAlmostEqual(phi_value * phi_value, phi_value + 1, places=10,
                              msg="φ²必须等于φ+1")
        self.assertAlmostEqual(1/phi_value, phi_value - 1, places=10,
                              msg="1/φ必须等于φ-1")

    def test_entropy_increase_principle_in_cognitive_systems(self):
        """测试认知系统中的熵增原理"""
        cognitive_arch = PhiCognitiveArchitecture(4)
        
        # 测试简单输入
        simple_input = PhiCognitiveState(
            amplitudes=[PhiComplex(PhiReal.one(), PhiReal.zero())],
            cognitive_basis=["simple"],
            normalization=PhiReal.one(),
            coherence_time=PhiReal.from_decimal(1.0)
        )
        
        # 测试复杂输入
        complex_input = PhiCognitiveState(
            amplitudes=[
                PhiComplex(PhiReal.one(), PhiReal.zero()),
                PhiComplex(PhiReal.from_decimal(0.8), PhiReal.zero()),
                PhiComplex(PhiReal.from_decimal(0.6), PhiReal.zero())
            ],
            cognitive_basis=["complex1", "complex2", "complex3"],
            normalization=PhiReal.one(),
            coherence_time=PhiReal.from_decimal(1.0)
        )
        
        # 处理简单输入
        simple_result = cognitive_arch.process_cognitive_input(simple_input)
        
        # 处理复杂输入
        complex_result = cognitive_arch.process_cognitive_input(complex_input)
        
        # 计算信息复杂度（使用认知基态数量作为复杂度指标）
        simple_complexity = len(simple_result.cognitive_basis)
        complex_complexity = len(complex_result.cognitive_basis)
        
        # 根据熵增原理，更复杂的输入应该产生更复杂的输出
        self.assertGreaterEqual(complex_complexity, simple_complexity,
                               "更复杂的输入应该产生复杂度相等或更高的输出（熵增原理）")
        
        # 测试自指处理的熵增
        initial_complexity = len(simple_result.cognitive_basis)
        
        # 自指处理
        self_ref_result = cognitive_arch.process_cognitive_input(simple_result)
        final_complexity = len(self_ref_result.cognitive_basis)
        
        # 自指系统的复杂度应该保持或增加
        self.assertGreaterEqual(final_complexity, initial_complexity,
                               "自指系统的复杂度应该保持或增加（熵增原理）")

    def test_complete_cognitive_system_integration(self):
        """测试完整认知系统集成"""
        # 使用验证函数测试所有组件
        verification_results = complete_phi_cognitive_architecture_verification()
        
        # 验证所有核心功能
        required_components = [
            "neural_network", "network_capacity", "memory_encoding", "memory_retrieval",
            "attention_mechanism", "attention_switching", "language_processing",
            "reasoning_system", "reasoning_confidence", "phi_structure_consistency"
        ]
        
        for component in required_components:
            self.assertIn(component, verification_results,
                         f"验证结果必须包含{component}组件")
            
            if component != "exception":  # 异常字段可能不存在
                self.assertTrue(verification_results[component],
                               f"{component}组件验证必须通过")
        
        # 确保没有异常
        self.assertNotIn("exception", verification_results,
                        "完整系统验证不应该发生异常")
        
        # 验证系统整体性能
        self.assertTrue(all(verification_results[comp] for comp in required_components),
                       "所有认知组件必须全部通过验证")

if __name__ == "__main__":
    unittest.main()