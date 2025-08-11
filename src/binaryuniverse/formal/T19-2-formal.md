# T19-2 φ-认知架构定理 - 形式化规范

## 类型定义

```python
from typing import Dict, List, Tuple, Optional, Callable, Union, Iterator
from dataclasses import dataclass
import numpy as np
import math
from enum import Enum
from phi_arithmetic import PhiReal, PhiComplex, PhiMatrix

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
    SENSORY = "sensory"      # 感觉记忆
    SHORT_TERM = "short_term"   # 短期记忆
    WORKING = "working"      # 工作记忆
    LONG_TERM = "long_term"  # 长期记忆

class AttentionType(Enum):
    """注意力类型"""
    FOCUS = "focus"          # 焦点注意力
    PERIPHERAL = "peripheral" # 边缘注意力
    BACKGROUND = "background" # 背景注意力

@dataclass
class ZeckendorfCognitive:
    """认知Zeckendorf编码"""
    fibonacci_coefficients: List[int]  # Fibonacci系数
    cognitive_meaning: str             # 认知意义
    no_consecutive_ones: bool = True   # no-11约束验证
    
    def __post_init__(self):
        """验证no-11约束"""
        for i in range(len(self.fibonacci_coefficients) - 1):
            if self.fibonacci_coefficients[i] == 1 and self.fibonacci_coefficients[i+1] == 1:
                raise ValueError(f"违反no-11约束: 位置{i}和{i+1}都为1")

@dataclass
class PhiCognitiveState:
    """φ-认知量子态"""
    amplitudes: List[PhiComplex]      # 量子振幅
    cognitive_basis: List[str]        # 认知基态标签
    normalization: PhiReal           # 归一化常数
    coherence_time: PhiReal          # 相干时间
    
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
    activation: PhiReal              # 激活值
    threshold: PhiReal               # 阈值
    connections: List[PhiReal]       # 连接权重
    fibonacci_index: int             # Fibonacci层级索引

@dataclass
class PhiMemoryTrace:
    """φ-记忆痕迹"""
    content: PhiCognitiveState       # 记忆内容
    strength: PhiReal                # 记忆强度
    timestamp: PhiReal               # 时间戳
    decay_rate: PhiReal              # 衰减率
    memory_type: MemoryType          # 记忆类型

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
                    # 自连接为0
                    connection = PhiComplex.zero()
                else:
                    # 连接强度按φ衰减
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
        
        # 设置输入层激活
        current_activations = input_data[:]
        
        # 逐层传播
        for layer_idx in range(1, self.num_layers):
            next_activations = []
            current_layer = self.layers[layer_idx]
            
            for neuron_idx, neuron in enumerate(current_layer):
                # 计算加权输入
                weighted_sum = PhiReal.zero()
                for prev_idx, prev_activation in enumerate(current_activations):
                    if prev_idx < len(neuron.connections):
                        weight = neuron.connections[prev_idx]
                        weighted_sum = weighted_sum + weight * prev_activation
                
                # 应用激活函数
                if weighted_sum.decimal_value > neuron.threshold.decimal_value:
                    activation = self.phi_activation(weighted_sum)
                else:
                    activation = PhiReal.zero()
                
                next_activations.append(activation)
            
            current_activations = next_activations
        
        # 记录激活历史
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
        
        # 记忆检索索引
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
    
    def _initialize_sensory_memory(self) -> List[PhiMemoryTrace]:
        """初始化感觉记忆"""
        capacity = self.fibonacci[1]  # F_1 = 1
        decay_time = PhiReal.from_decimal(0.5)  # 0.5秒
        return [None] * capacity
    
    def _initialize_short_term_memory(self) -> List[PhiMemoryTrace]:
        """初始化短期记忆"""
        capacity = self.fibonacci[5]  # F_5 = 5
        decay_time = PhiReal.from_decimal(0.5) * self.phi  # 0.81秒
        return [None] * capacity
    
    def _initialize_working_memory(self) -> List[PhiMemoryTrace]:
        """初始化工作记忆"""
        capacity = self.fibonacci[6]  # F_6 = 8
        decay_time = PhiReal.from_decimal(0.5) * (self.phi ** 2)  # 1.31秒
        return [None] * capacity
    
    def _initialize_long_term_memory(self) -> List[PhiMemoryTrace]:
        """初始化长期记忆"""
        capacity = self.fibonacci[13]  # F_13 = 233
        decay_time = PhiReal.from_decimal(0.5) * (self.phi ** 3)  # 2.12秒
        return [None] * capacity
    
    def encode_memory(self, content: PhiCognitiveState, memory_type: MemoryType, 
                     current_time: PhiReal) -> bool:
        """编码记忆"""
        memory_layer = self.memory_layers[memory_type]
        
        # 创建记忆痕迹
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
        
        # 存储记忆
        if empty_slot is not None:
            memory_layer[empty_slot] = trace
        else:
            memory_layer[weakest_slot] = trace
        
        # 更新检索索引
        content_key = hash(str(content.cognitive_basis))
        if content_key not in self.retrieval_index:
            self.retrieval_index[content_key] = []
        self.retrieval_index[content_key].append((memory_type, len(memory_layer) - 1))
        
        return True
    
    def retrieve_memory(self, cue: PhiCognitiveState, current_time: PhiReal) -> Optional[PhiMemoryTrace]:
        """检索记忆"""
        best_match = None
        best_similarity = PhiReal.zero()
        
        # 在所有记忆层中搜索
        for memory_type, memory_layer in self.memory_layers.items():
            for trace in memory_layer:
                if trace is None:
                    continue
                
                # 计算记忆衰减
                time_diff = current_time - trace.timestamp
                decay_factor = (-time_diff / trace.decay_rate).exp()
                current_strength = trace.strength * decay_factor
                
                # 如果记忆太弱，跳过
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
            # 计算复数内积的实部
            real_part = cue.amplitudes[i].real * memory.amplitudes[i].real + \
                       cue.amplitudes[i].imag * memory.amplitudes[i].imag
            dot_product = dot_product + real_part
            
            # 计算模长
            cue_norm_sq = cue.amplitudes[i].real * cue.amplitudes[i].real + \
                         cue.amplitudes[i].imag * cue.amplitudes[i].imag
            memory_norm_sq = memory.amplitudes[i].real * memory.amplitudes[i].real + \
                           memory.amplitudes[i].imag * memory.amplitudes[i].imag
            
            cue_norm = cue_norm + cue_norm_sq
            memory_norm = memory_norm + memory_norm_sq
        
        # 计算余弦相似度
        if cue_norm.decimal_value > 0 and memory_norm.decimal_value > 0:
            similarity = dot_product / (cue_norm.sqrt() * memory_norm.sqrt())
            return PhiReal.from_decimal(max(0, similarity.decimal_value))
        else:
            return PhiReal.zero()
    
    def decay_memories(self, current_time: PhiReal):
        """记忆衰减处理"""
        for memory_type, memory_layer in self.memory_layers.items():
            for i, trace in enumerate(memory_layer):
                if trace is None:
                    continue
                
                # 计算衰减
                time_diff = current_time - trace.timestamp
                decay_factor = (-time_diff / trace.decay_rate).exp()
                new_strength = trace.strength * decay_factor
                
                # 如果强度太低，删除记忆
                if new_strength.decimal_value < 0.001:
                    memory_layer[i] = None
                else:
                    trace.strength = new_strength

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
        
        # 初始化注意力权重矩阵
        self.query_weights = self._initialize_weights(input_dim, input_dim)
        self.key_weights = self._initialize_weights(input_dim, input_dim)
        self.value_weights = self._initialize_weights(input_dim, input_dim)
        
        # 注意力历史
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
    
    def _initialize_weights(self, input_dim: int, output_dim: int) -> PhiMatrix:
        """初始化权重矩阵"""
        weights_data = []
        
        for i in range(output_dim):
            row = []
            for j in range(input_dim):
                # 权重按φ分布初始化
                weight_val = PhiReal.one() / (self.phi ** abs(i - j)) if i != j else PhiReal.one()
                weight = PhiComplex(weight_val, PhiReal.zero())
                row.append(weight)
            weights_data.append(row)
        
        return PhiMatrix(weights_data)
    
    def phi_softmax(self, scores: List[PhiReal]) -> List[PhiReal]:
        """φ-softmax函数"""
        # 计算 φ^x
        phi_scores = []
        for score in scores:
            phi_score = self.phi ** score.decimal_value
            phi_scores.append(PhiReal.from_decimal(phi_score))
        
        # 计算归一化
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
        
        # 计算Query, Key, Value
        queries = self._compute_qkv(inputs, self.query_weights)
        keys = self._compute_qkv(inputs, self.key_weights)
        values = self._compute_qkv(inputs, self.value_weights)
        
        # 计算注意力分数
        attention_scores = []
        for i in range(num_inputs):
            scores_for_i = []
            for j in range(num_inputs):
                score = self._compute_dot_product(queries[i], keys[j])
                # 除以sqrt(d_k/φ)进行缩放
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
            weighted_sum = self._compute_weighted_sum(values, attention_weights[i])
            outputs.append(weighted_sum)
        
        # 计算平均注意力权重用于返回
        avg_weights = []
        for j in range(num_inputs):
            total_weight = PhiReal.zero()
            for i in range(num_inputs):
                total_weight = total_weight + attention_weights[i][j]
            avg_weight = total_weight / PhiReal.from_decimal(num_inputs)
            avg_weights.append(avg_weight)
        
        # 记录注意力历史
        self.attention_history.append(attention_weights)
        
        return outputs, avg_weights
    
    def _compute_qkv(self, inputs: List[PhiCognitiveState], weight_matrix: PhiMatrix) -> List[PhiCognitiveState]:
        """计算Query/Key/Value变换"""
        results = []
        
        for cognitive_state in inputs:
            # 将认知状态转换为向量
            input_vector = cognitive_state.amplitudes
            
            # 矩阵乘法
            output_amplitudes = []
            for i in range(len(weight_matrix.data)):
                weighted_sum = PhiComplex.zero()
                for j in range(min(len(input_vector), len(weight_matrix.data[i]))):
                    weighted_sum = weighted_sum + weight_matrix.data[i][j] * input_vector[j]
                output_amplitudes.append(weighted_sum)
            
            # 创建输出认知状态
            output_state = PhiCognitiveState(
                amplitudes=output_amplitudes,
                cognitive_basis=cognitive_state.cognitive_basis[:],
                normalization=cognitive_state.normalization,
                coherence_time=cognitive_state.coherence_time
            )
            results.append(output_state)
        
        return results
    
    def _compute_dot_product(self, state1: PhiCognitiveState, state2: PhiCognitiveState) -> PhiReal:
        """计算两个认知状态的内积"""
        if len(state1.amplitudes) != len(state2.amplitudes):
            return PhiReal.zero()
        
        dot_product = PhiReal.zero()
        for i in range(len(state1.amplitudes)):
            # 复数内积的实部
            real_part = state1.amplitudes[i].real * state2.amplitudes[i].real + \
                       state1.amplitudes[i].imag * state2.amplitudes[i].imag
            dot_product = dot_product + real_part
        
        return dot_product
    
    def _compute_weighted_sum(self, values: List[PhiCognitiveState], weights: List[PhiReal]) -> PhiCognitiveState:
        """计算加权和"""
        if not values or not weights:
            return PhiCognitiveState([], [], PhiReal.zero(), PhiReal.zero())
        
        # 初始化结果
        result_amplitudes = [PhiComplex.zero() for _ in range(len(values[0].amplitudes))]
        result_basis = values[0].cognitive_basis[:]
        
        # 计算加权和
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
        
        # 语法参数
        self.max_depth = self.fibonacci[8]  # F_8 = 21
        self.semantic_dim = int(self.fibonacci[10] * math.log(self.phi.decimal_value, 2))  # F_10 * log2(φ)
        
        # 初始化词汇表和语义空间
        self.vocabulary = self._initialize_vocabulary()
        self.semantic_space = self._initialize_semantic_space()
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
        
        # 生成测试词汇
        test_words = ["the", "cat", "dog", "runs", "jumps", "quickly", "slowly", "red", "blue", "house"]
        
        for i, word in enumerate(test_words[:min(self.vocab_size, len(test_words))]):
            # 将词汇编码为Zeckendorf表示
            zeck_encoding = self._word_to_zeckendorf(word, i)
            vocab[word] = zeck_encoding
        
        return vocab
    
    def _word_to_zeckendorf(self, word: str, index: int) -> ZeckendorfCognitive:
        """将词汇转换为Zeckendorf编码"""
        # 使用词汇索引生成Zeckendorf表示
        coeffs = self._to_zeckendorf(index + 1)
        
        return ZeckendorfCognitive(
            fibonacci_coefficients=coeffs,
            cognitive_meaning=f"word: {word}"
        )
    
    def _to_zeckendorf(self, n: int) -> List[int]:
        """将整数转换为Zeckendorf表示"""
        if n == 0:
            return [0]
        
        coeffs = []
        fib_index = 0
        
        # 找到最大的不超过n的Fibonacci数
        while fib_index < len(self.fibonacci) and self.fibonacci[fib_index] <= n:
            fib_index += 1
        fib_index -= 1
        
        # 贪心算法构造Zeckendorf表示
        coeffs = [0] * (fib_index + 1)
        remaining = n
        
        for i in range(fib_index, -1, -1):
            if remaining >= self.fibonacci[i]:
                coeffs[i] = 1
                remaining -= self.fibonacci[i]
        
        return coeffs
    
    def _initialize_semantic_space(self) -> PhiMatrix:
        """初始化语义空间"""
        semantic_data = []
        
        for i in range(self.semantic_dim):
            row = []
            for j in range(self.vocab_size):
                # 语义权重按φ分布
                weight_val = PhiReal.one() / (self.phi ** (abs(i - j) % 10))
                semantic_weight = PhiComplex(weight_val, PhiReal.zero())
                row.append(semantic_weight)
            semantic_data.append(row)
        
        return PhiMatrix(semantic_data)
    
    def _initialize_syntax_rules(self) -> Dict[str, List[str]]:
        """初始化语法规则"""
        return {
            "S": ["NP VP"],           # 句子 -> 名词短语 + 动词短语
            "NP": ["Det N", "N"],     # 名词短语 -> 限定词 + 名词 或 名词
            "VP": ["V", "V NP", "V Adv"],  # 动词短语 -> 动词 或 动词 + 名词短语 或 动词 + 副词
            "Det": ["the"],           # 限定词
            "N": ["cat", "dog", "house"],  # 名词
            "V": ["runs", "jumps"],   # 动词
            "Adv": ["quickly", "slowly"]  # 副词
        }
    
    def tokenize(self, sentence: str) -> List[str]:
        """词法分析"""
        # 简单的词法分析：按空格分割
        tokens = sentence.lower().split()
        return tokens
    
    def parse_sentence(self, tokens: List[str]) -> Optional[Dict]:
        """句法解析 - 递归下降解析器"""
        if not tokens:
            return None
        
        # 使用φ-递归解析
        parse_tree = self._recursive_parse("S", tokens, 0, 0)
        
        return parse_tree
    
    def _recursive_parse(self, symbol: str, tokens: List[str], position: int, depth: int) -> Optional[Dict]:
        """φ-递归解析"""
        if depth > self.max_depth:
            return None
        
        if position >= len(tokens):
            return None
        
        # 如果是终结符
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
        
        # 如果是非终结符，尝试所有规则
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
    
    def extract_semantics(self, parse_tree: Dict) -> PhiCognitiveState:
        """语义提取"""
        if parse_tree is None:
            return PhiCognitiveState([], [], PhiReal.zero(), PhiReal.zero())
        
        # 递归提取语义
        semantic_vector = self._extract_semantic_vector(parse_tree)
        
        # 创建认知状态
        amplitudes = []
        basis_labels = []
        
        for i, component in enumerate(semantic_vector):
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
    
    def _extract_semantic_vector(self, node: Dict) -> List[PhiReal]:
        """递归提取语义向量"""
        semantic_vector = [PhiReal.zero() for _ in range(self.semantic_dim)]
        
        if "value" in node:
            # 叶子节点 - 查找词汇语义
            word = node["value"]
            if word in self.vocabulary:
                word_encoding = self.vocabulary[word]
                # 将Zeckendorf编码映射到语义空间
                for i, coeff in enumerate(word_encoding.fibonacci_coefficients):
                    if i < self.semantic_dim and coeff == 1:
                        semantic_vector[i] = semantic_vector[i] + node["phi_weight"]
        
        if "children" in node:
            # 内部节点 - 组合子节点语义
            for child in node["children"]:
                child_semantics = self._extract_semantic_vector(child)
                for i in range(len(semantic_vector)):
                    if i < len(child_semantics):
                        semantic_vector[i] = semantic_vector[i] + child_semantics[i] / self.phi
        
        return semantic_vector
    
    def understand_sentence(self, sentence: str) -> PhiCognitiveState:
        """完整句子理解流程"""
        # 1. 词法分析
        tokens = self.tokenize(sentence)
        
        # 2. 句法解析
        parse_tree = self.parse_sentence(tokens)
        
        # 3. 语义提取
        semantics = self.extract_semantics(parse_tree)
        
        return semantics

class PhiReasoningSystem:
    """φ-推理系统 - 完整逻辑推理引擎"""
    
    def __init__(self, max_proof_depth: int = 8):
        """初始化φ-推理系统"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.max_proof_depth = max_proof_depth
        self.fibonacci = self._generate_fibonacci(max_proof_depth + 5)
        
        # 推理规则权重
        self.inference_rules = {
            "modus_ponens": self.phi ** 0,      # 假言推理，权重φ^0
            "modus_tollens": self.phi ** (-1),  # 拒取推理，权重φ^(-1)
            "hypothetical_syllogism": self.phi ** (-1),  # 假言三段论
            "disjunctive_syllogism": self.phi ** (-1),   # 析取三段论
            "induction": self.phi ** (-2)       # 归纳推理，权重φ^(-2)
        }
        
        # 知识库
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
        # 检查缓存
        if goal in self.proof_cache:
            return self.proof_cache[goal]
        
        # 初始化证明搜索
        proof_tree = self._search_proof(goal, 0, [])
        
        if proof_tree is not None:
            success = True
            confidence = self._calculate_proof_confidence(proof_tree)
            proof_steps = self._extract_proof_steps(proof_tree)
        else:
            success = False
            confidence = PhiReal.zero()
            proof_steps = []
        
        # 缓存结果
        result = (success, proof_steps, confidence)
        self.proof_cache[goal] = result
        
        return result
    
    def _search_proof(self, goal: str, depth: int, used_facts: List[str]) -> Optional[Dict]:
        """搜索证明"""
        if depth > self.max_proof_depth:
            return None
        
        # 检查是否直接在知识库中
        for knowledge in self.knowledge_base:
            if knowledge["proposition"] == goal and knowledge not in used_facts:
                return {
                    "type": "fact",
                    "goal": goal,
                    "knowledge": knowledge,
                    "depth": depth,
                    "confidence": knowledge["certainty"]
                }
        
        # 尝试各种推理规则
        best_proof = None
        best_confidence = PhiReal.zero()
        
        # 假言推理：如果有 P -> Q 和 P，则可以推出 Q
        modus_ponens_proof = self._try_modus_ponens(goal, depth, used_facts)
        if modus_ponens_proof is not None:
            confidence = self._calculate_proof_confidence(modus_ponens_proof)
            if confidence.decimal_value > best_confidence.decimal_value:
                best_proof = modus_ponens_proof
                best_confidence = confidence
        
        # 拒取推理：如果有 P -> Q 和 ¬Q，则可以推出 ¬P
        modus_tollens_proof = self._try_modus_tollens(goal, depth, used_facts)
        if modus_tollens_proof is not None:
            confidence = self._calculate_proof_confidence(modus_tollens_proof)
            if confidence.decimal_value > best_confidence.decimal_value:
                best_proof = modus_tollens_proof
                best_confidence = confidence
        
        # 假言三段论：如果有 P -> Q 和 Q -> R，则可以推出 P -> R
        hypothetical_proof = self._try_hypothetical_syllogism(goal, depth, used_facts)
        if hypothetical_proof is not None:
            confidence = self._calculate_proof_confidence(hypothetical_proof)
            if confidence.decimal_value > best_confidence.decimal_value:
                best_proof = hypothetical_proof
                best_confidence = confidence
        
        return best_proof
    
    def _try_modus_ponens(self, goal: str, depth: int, used_facts: List[str]) -> Optional[Dict]:
        """尝试假言推理"""
        # 寻找形如 "P -> goal" 的规则
        for knowledge in self.knowledge_base:
            if knowledge in used_facts:
                continue
                
            prop = knowledge["proposition"]
            if " -> " in prop and prop.endswith(" -> " + goal):
                antecedent = prop.split(" -> ")[0]
                
                # 尝试证明前件
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
    
    def _try_modus_tollens(self, goal: str, depth: int, used_facts: List[str]) -> Optional[Dict]:
        """尝试拒取推理"""
        # 寻找形如 "¬goal" 的目标，以及相应的条件语句
        if goal.startswith("¬"):
            positive_goal = goal[1:]  # 去掉否定符号
            
            for knowledge in self.knowledge_base:
                if knowledge in used_facts:
                    continue
                    
                prop = knowledge["proposition"]
                if " -> " in prop and prop.endswith(" -> " + positive_goal):
                    antecedent = prop.split(" -> ")[0]
                    
                    # 尝试证明前件的否定形式不成立（即证明前件为假）
                    negated_antecedent = "¬" + antecedent
                    neg_proof = self._search_proof(negated_antecedent, depth + 1, used_facts + [knowledge])
                    
                    if neg_proof is not None:
                        return {
                            "type": "modus_tollens",
                            "goal": goal,
                            "rule": knowledge,
                            "negated_antecedent_proof": neg_proof,
                            "depth": depth,
                            "rule_weight": self.inference_rules["modus_tollens"]
                        }
        
        return None
    
    def _try_hypothetical_syllogism(self, goal: str, depth: int, used_facts: List[str]) -> Optional[Dict]:
        """尝试假言三段论"""
        if " -> " not in goal:
            return None
        
        # 解析目标：P -> R
        parts = goal.split(" -> ")
        if len(parts) != 2:
            return None
        
        P, R = parts[0], parts[1]
        
        # 寻找中间项Q，使得有 P -> Q 和 Q -> R
        for knowledge1 in self.knowledge_base:
            if knowledge1 in used_facts:
                continue
                
            prop1 = knowledge1["proposition"]
            if " -> " in prop1 and prop1.startswith(P + " -> "):
                Q = prop1.split(" -> ")[1]
                
                # 寻找 Q -> R
                for knowledge2 in self.knowledge_base:
                    if knowledge2 in used_facts or knowledge2 == knowledge1:
                        continue
                        
                    prop2 = knowledge2["proposition"]
                    if prop2 == Q + " -> " + R:
                        return {
                            "type": "hypothetical_syllogism",
                            "goal": goal,
                            "rule1": knowledge1,  # P -> Q
                            "rule2": knowledge2,  # Q -> R
                            "intermediate": Q,
                            "depth": depth,
                            "rule_weight": self.inference_rules["hypothetical_syllogism"]
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
            
            # 组合置信度
            combined = rule_confidence * antecedent_confidence * rule_weight
            return combined
        
        elif proof_tree["type"] == "modus_tollens":
            rule_confidence = proof_tree["rule"]["certainty"]
            neg_confidence = self._calculate_proof_confidence(proof_tree["negated_antecedent_proof"])
            rule_weight = proof_tree["rule_weight"]
            
            combined = rule_confidence * neg_confidence * rule_weight
            return combined
        
        elif proof_tree["type"] == "hypothetical_syllogism":
            rule1_confidence = proof_tree["rule1"]["certainty"]
            rule2_confidence = proof_tree["rule2"]["certainty"]
            rule_weight = proof_tree["rule_weight"]
            
            combined = rule1_confidence * rule2_confidence * rule_weight
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
            # 先添加前件的证明步骤
            antecedent_steps = self._extract_proof_steps(proof_tree["antecedent_proof"])
            steps.extend(antecedent_steps)
            
            # 添加假言推理步骤
            steps.append({
                "step": f"By Modus Ponens from '{proof_tree['rule']['proposition']}' and previous: {proof_tree['goal']}",
                "confidence": self._calculate_proof_confidence(proof_tree).decimal_value
            })
        
        elif proof_tree["type"] == "modus_tollens":
            neg_steps = self._extract_proof_steps(proof_tree["negated_antecedent_proof"])
            steps.extend(neg_steps)
            
            steps.append({
                "step": f"By Modus Tollens from '{proof_tree['rule']['proposition']}' and previous: {proof_tree['goal']}",
                "confidence": self._calculate_proof_confidence(proof_tree).decimal_value
            })
        
        elif proof_tree["type"] == "hypothetical_syllogism":
            steps.append({
                "step": f"By Hypothetical Syllogism from '{proof_tree['rule1']['proposition']}' and '{proof_tree['rule2']['proposition']}': {proof_tree['goal']}",
                "confidence": self._calculate_proof_confidence(proof_tree).decimal_value
            })
        
        return steps
    
    def calculate_reasoning_complexity(self, proof_steps: List[Dict]) -> PhiReal:
        """计算推理复杂度"""
        if not proof_steps:
            return PhiReal.zero()
        
        n = len(proof_steps)
        complexity = self.phi ** n
        return complexity

def verify_cognitive_self_reference_property(cognitive_arch: 'PhiCognitiveArchitecture') -> bool:
    """验证认知系统的自指性质：C = C[C]"""
    
    # 测试输入
    test_input = PhiCognitiveState(
        amplitudes=[PhiComplex(PhiReal.one(), PhiReal.zero())],
        cognitive_basis=["test_concept"],
        normalization=PhiReal.one(),
        coherence_time=PhiReal.from_decimal(1.0)
    )
    
    # 第一次认知处理：C
    result1 = cognitive_arch.process_cognitive_input(test_input)
    
    # 第二次认知处理：C[C] - 将结果再次输入系统
    if result1 is not None:
        result2 = cognitive_arch.process_cognitive_input(result1)
    else:
        return False
    
    # 验证自指性质
    # 1. 系统必须能够处理自己的输出
    processing_successful = result2 is not None and len(result2.amplitudes) > 0
    
    # 2. 相干时间必须保持或增加（根据唯一公理）
    coherence_maintained = result2 is not None and \
                          result2.coherence_time.decimal_value >= result1.coherence_time.decimal_value * 0.9
    
    # 3. 认知复杂度应该增加
    complexity_increased = result2 is not None and \
                          len(result2.amplitudes) >= len(result1.amplitudes)
    
    return processing_successful and coherence_maintained and complexity_increased

# 完整的φ-认知架构验证函数
def complete_phi_cognitive_architecture_verification() -> Dict[str, bool]:
    """完整验证φ-认知架构系统的所有核心性质"""
    
    results = {}
    
    try:
        # 1. 验证φ-神经网络架构
        neural_net = PhiNeuralNetwork(5)
        test_input = [PhiReal.from_decimal(0.5), PhiReal.one(), PhiReal.from_decimal(0.3)]
        
        # 扩展输入到匹配第一层大小
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
        
        # 测试记忆编码
        encode_success = memory_system.encode_memory(test_memory, MemoryType.WORKING, PhiReal.zero())
        results["memory_encoding"] = encode_success
        
        # 测试记忆检索
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
        
        # 添加测试知识
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

class PhiCognitiveArchitecture:
    """φ-认知架构主系统 - 完整自指认知实现"""
    
    def __init__(self, input_dim: int = 8):
        """初始化φ-认知架构"""
        self.phi = PhiReal.from_decimal((1 + math.sqrt(5)) / 2)
        self.input_dim = input_dim
        
        # 初始化12个认知子系统
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
        
        # 检索相关记忆
        retrieved_memory = self.memory_system.retrieve_memory(attended_state, current_time)
        
        # 整合当前输入和记忆
        if retrieved_memory is not None:
            # 简单的记忆-输入整合
            integrated_amplitudes = []
            
            max_len = max(len(attended_state.amplitudes), len(retrieved_memory.content.amplitudes))
            for i in range(max_len):
                current_amp = attended_state.amplitudes[i] if i < len(attended_state.amplitudes) else PhiComplex.zero()
                memory_amp = retrieved_memory.content.amplitudes[i] if i < len(retrieved_memory.content.amplitudes) else PhiComplex.zero()
                
                # φ-加权整合
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
        
        # 5. 返回处理后的认知状态
        return integrated_state
```

<system-reminder>
Whenever you write files, double-check the content is consistent with what you've already written. The user expects consistency across all T19-2 materials.
</system-reminder>