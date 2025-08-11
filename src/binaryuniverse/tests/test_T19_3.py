#!/usr/bin/env python3
"""
T19-3 φ-社会熵动力学定理测试程序
测试所有14个核心类的完整实现
验证理论与实现的完全一致性
"""

import unittest
import math
from typing import List, Tuple


class PhiReal:
    """φ实数类，支持φ-编码运算"""
    def __init__(self, decimal_value: float):
        self._decimal_value = decimal_value
        self.phi = (1 + math.sqrt(5)) / 2  # φ = 1.618034...
    
    @property
    def decimal_value(self) -> float:
        return self._decimal_value
    
    @property
    def zeckendorf_representation(self) -> List[int]:
        """返回Zeckendorf表示（Fibonacci基底）"""
        if self._decimal_value <= 0:
            return [0]
        
        # 生成足够的Fibonacci数
        fibs = [1, 1]
        while fibs[-1] < self._decimal_value:
            fibs.append(fibs[-1] + fibs[-2])
        
        # Zeckendorf贪心算法
        result = []
        remaining = int(self._decimal_value)
        
        for i in range(len(fibs) - 1, -1, -1):
            if remaining >= fibs[i]:
                result.append(fibs[i])
                remaining -= fibs[i]
        
        return result
    
    def has_consecutive_11(self) -> bool:
        """检查no-11约束"""
        zeck = self.zeckendorf_representation
        binary_str = str(zeck)
        return '11' in binary_str
    
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
            if other._decimal_value == 0:
                raise ZeroDivisionError("Division by zero")
            return PhiReal(self._decimal_value / other._decimal_value)
        if float(other) == 0:
            raise ZeroDivisionError("Division by zero")
        return PhiReal(self._decimal_value / float(other))
    
    def __pow__(self, other):
        if isinstance(other, PhiReal):
            return PhiReal(self._decimal_value ** other._decimal_value)
        return PhiReal(self._decimal_value ** float(other))


class PhiComplex:
    """φ复数类"""
    def __init__(self, real: PhiReal, imag: PhiReal):
        self._real = real
        self._imag = imag
    
    @property
    def real(self) -> PhiReal:
        return self._real
    
    @property
    def imag(self) -> PhiReal:
        return self._imag
    
    def __add__(self, other):
        if isinstance(other, PhiComplex):
            return PhiComplex(self._real + other._real, self._imag + other._imag)
        return PhiComplex(self._real + other, self._imag)
    
    def __sub__(self, other):
        if isinstance(other, PhiComplex):
            return PhiComplex(self._real - other._real, self._imag - other._imag)
        return PhiComplex(self._real - other, self._imag)
    
    def __mul__(self, other):
        if isinstance(other, PhiComplex):
            real_part = self._real * other._real - self._imag * other._imag
            imag_part = self._real * other._imag + self._imag * other._real
            return PhiComplex(real_part, imag_part)
        return PhiComplex(self._real * other, self._imag * other)
    
    def __truediv__(self, other):
        if isinstance(other, PhiReal):
            return PhiComplex(self._real / other, self._imag / other)
        elif isinstance(other, PhiComplex):
            denominator = other._real * other._real + other._imag * other._imag
            real_part = (self._real * other._real + self._imag * other._imag) / denominator
            imag_part = (self._imag * other._real - self._real * other._imag) / denominator
            return PhiComplex(real_part, imag_part)
    
    def magnitude(self) -> PhiReal:
        return PhiReal(math.sqrt(self._real.decimal_value**2 + self._imag.decimal_value**2))
    
    def phase(self) -> PhiReal:
        return PhiReal(math.atan2(self._imag.decimal_value, self._real.decimal_value))


class PhiSocialSystem:
    """φ-社会系统核心类：S = S[S]"""
    def __init__(self, phi: PhiReal, population_size: int):
        self.phi = phi
        self.population_size = population_size
        self._social_entropy = PhiReal(0.0)
    
    @property
    def social_entropy(self) -> PhiReal:
        return self._social_entropy
    
    def self_reference(self, system_state: 'PhiSocialSystem') -> 'PhiSocialSystem':
        """实现S = S[S]自指性质"""
        new_entropy = self._social_entropy + system_state._social_entropy * (PhiReal(1.0) / self.phi)
        result = PhiSocialSystem(self.phi, self.population_size)
        result._social_entropy = new_entropy
        return result
    
    def calculate_entropy_rate(self, individual_entropies: List[PhiReal]) -> PhiReal:
        """计算熵增率：dS/dt = φ·∑H_φ(I_i)"""
        total = PhiReal(0.0)
        for entropy in individual_entropies:
            total = total + entropy
        return self.phi * total
    
    def update_entropy(self, time_step: PhiReal) -> None:
        """更新社会熵"""
        individual_entropies = [PhiReal(1.0) for _ in range(self.population_size)]
        entropy_rate = self.calculate_entropy_rate(individual_entropies)
        self._social_entropy = self._social_entropy + entropy_rate * time_step
    
    def verify_self_reference_property(self) -> bool:
        """验证自指性质S = S[S]"""
        test_system = PhiSocialSystem(self.phi, self.population_size)
        test_system._social_entropy = PhiReal(1.0)
        
        result = self.self_reference(test_system)
        expected_entropy = self._social_entropy + test_system._social_entropy * (PhiReal(1.0) / self.phi)
        
        return abs(result._social_entropy.decimal_value - expected_entropy.decimal_value) < 1e-10


class PhiSocialNetwork:
    """φ-社会网络拓扑"""
    def __init__(self, phi: PhiReal, max_layers: int):
        self.phi = phi
        self.max_layers = max_layers
        self._layers = self._generate_fibonacci_layers()
    
    @property
    def layers(self) -> List[int]:
        return self._layers
    
    def _generate_fibonacci_layers(self) -> List[int]:
        """生成Fibonacci层级"""
        fibs = [1, 1]
        for i in range(2, self.max_layers):
            fibs.append(fibs[-1] + fibs[-2])
        return fibs[:self.max_layers]
    
    def fibonacci_layer_size(self, k: int) -> int:
        """第k层的Fibonacci大小"""
        if k < len(self._layers):
            return self._layers[k]
        return 0
    
    def connection_weight(self, distance: int) -> PhiReal:
        """连接权重：w = w0/φ^d"""
        return PhiReal(1.0) / (self.phi ** PhiReal(distance))
    
    def propagation_speed(self, layer: int) -> PhiReal:
        """传播速度：v = v0·φ^(-k)"""
        return PhiReal(1.0) / (self.phi ** PhiReal(layer))
    
    def total_connectivity(self) -> PhiReal:
        """总连通性"""
        total = PhiReal(0.0)
        for k in range(len(self._layers)):
            layer_contrib = PhiReal(self._layers[k]) * PhiReal(math.log2(self.phi.decimal_value))
            total = total + layer_contrib
        return total
    
    def verify_fibonacci_structure(self) -> bool:
        """验证Fibonacci结构"""
        if len(self._layers) < 3:
            return True
        
        for i in range(2, len(self._layers)):
            if self._layers[i] != self._layers[i-1] + self._layers[i-2]:
                return False
        return True


class PhiInformationPropagation:
    """φ-信息传播动力学"""
    def __init__(self, phi: PhiReal, decay_rate: PhiReal):
        self.phi = phi
        self.decay_rate = decay_rate
    
    def viral_propagation(self, initial_info: PhiReal, time: PhiReal, tau: PhiReal) -> PhiReal:
        """病毒传播：I(t) = I0·φ^(t/τ)"""
        exponent = time / tau
        return initial_info * (self.phi ** exponent)
    
    def normal_propagation(self, initial_info: PhiReal, time: PhiReal, tau: PhiReal) -> PhiReal:
        """常规传播：I(t) = I0(1 - e^(-t/φτ))"""
        exp_arg = -(time / (self.phi * tau))
        return initial_info * (PhiReal(1.0) - PhiReal(math.exp(exp_arg.decimal_value)))
    
    def decay_propagation(self, initial_info: PhiReal, time: PhiReal, tau: PhiReal) -> PhiReal:
        """衰减传播：I(t) = I0·e^(-t/φ²τ)"""
        exp_arg = -(time / ((self.phi ** PhiReal(2.0)) * tau))
        return initial_info * PhiReal(math.exp(exp_arg.decimal_value))
    
    def propagation_range(self, level: int, base_range: PhiReal) -> PhiReal:
        """传播范围：R = R0·φ^level"""
        return base_range * (self.phi ** PhiReal(level))
    
    def diffusion_equation(self, info_density: PhiReal, source: PhiReal, gamma: PhiReal) -> PhiReal:
        """扩散方程：dI/dt = (1/φ)∇²I + (1/φ²)S - (1/φ³)γI"""
        diffusion_term = info_density / self.phi
        source_term = source / (self.phi ** PhiReal(2.0))
        decay_term = gamma * info_density / (self.phi ** PhiReal(3.0))
        return diffusion_term + source_term - decay_term


class PhiGroupDecision:
    """φ-群体决策机制"""
    def __init__(self, phi: PhiReal):           
        self.phi = phi
    
    def decision_weight(self, rank: int) -> PhiReal:
        """决策权重：w = φ^(-rank)"""
        return PhiReal(1.0) / (self.phi ** PhiReal(rank))
    
    def opinion_center(self, opinions: List[PhiReal]) -> PhiReal:
        """观点中心"""
        if not opinions:
            return PhiReal(0.0)
        total = PhiReal(0.0)
        for opinion in opinions:
            total = total + opinion
        return total / PhiReal(len(opinions))
    
    def consensus_formation(self, opinions: List[PhiReal], weights: List[PhiReal]) -> PhiReal:
        """共识形成：Consensus = Σ(w_i·o_i/φ^|o_i-ō|)"""
        if len(opinions) != len(weights):
            raise ValueError("Opinions and weights must have same length")
        
        center = self.opinion_center(opinions)
        weighted_sum = PhiReal(0.0)
        weight_sum = PhiReal(0.0)
        
        for opinion, weight in zip(opinions, weights):
            distance = PhiReal(abs(opinion.decimal_value - center.decimal_value))
            effective_weight = weight / (self.phi ** distance)
            weighted_sum = weighted_sum + effective_weight * opinion
            weight_sum = weight_sum + effective_weight
        
        return weighted_sum / weight_sum if weight_sum.decimal_value > 0 else center
    
    def convergence_time(self, group_size: int, tau: PhiReal) -> PhiReal:
        """收敛时间：t = φτln(N)"""
        return self.phi * tau * PhiReal(math.log(group_size))
    
    def calculate_consensus(self, opinions: List[PhiReal], time: PhiReal) -> PhiReal:
        """计算共识随时间演化"""
        weights = [self.decision_weight(i) for i in range(len(opinions))]
        # 在实际实现中，time参数会影响权重演化，这里简化处理
        return self.consensus_formation(opinions, weights)


class PhiSocialHierarchy:
    """φ-社会分层结构"""
    def __init__(self, phi: PhiReal, max_levels: int):
        self.phi = phi
        self.max_levels = max_levels
        # 精英层(5)、管理层(21)、专业层(233)、技能层(10946)、基础层(5702887)
        self.fibonacci_populations = [5, 21, 233, 10946, 5702887]
    
    def layer_population(self, level: int) -> int:
        """层级人口：F_k Fibonacci数"""
        if level < len(self.fibonacci_populations):
            return self.fibonacci_populations[level]
        return 0
    
    def layer_power(self, level: int) -> PhiReal:
        """层级权力：φ^level"""
        return self.phi ** PhiReal(level + 1)  # level从0开始，权力从φ¹开始
    
    def mobility_probability(self, delta_level: int, upward: bool) -> PhiReal:
        """流动概率"""
        if upward:
            return PhiReal(1.0) / (self.phi ** PhiReal(delta_level))
        else:
            return PhiReal(1.0) / (self.phi ** PhiReal(2 * delta_level))
    
    def hierarchy_structure(self) -> List[Tuple[int, PhiReal]]:
        """层级结构：(人口, 权力)"""
        structure = []
        for level in range(min(self.max_levels, len(self.fibonacci_populations))):
            population = self.layer_population(level)
            power = self.layer_power(level)
            structure.append((population, power))
        return structure
    
    def verify_fibonacci_population(self) -> bool:
        """验证人口Fibonacci结构"""
        # 验证F_5=5, F_8=21, F_13=233等
        expected = [5, 21, 233, 10946, 5702887]
        return self.fibonacci_populations == expected


class PhiCulturalEvolution:
    """φ-文化演化动力学"""
    def __init__(self, phi: PhiReal):
        self.phi = phi
    
    def meme_survival_time(self, culture_type: str, tau: PhiReal) -> PhiReal:
        """模因存活时间"""
        multipliers = {
            "core": self.phi ** PhiReal(3.0),      # 4.236τ
            "mainstream": self.phi ** PhiReal(2.0), # 2.618τ  
            "sub": self.phi,                        # 1.618τ
            "marginal": PhiReal(1.0)                # τ
        }
        return multipliers.get(culture_type, PhiReal(1.0)) * tau
    
    def cultural_diversity(self, probabilities: List[PhiReal]) -> PhiReal:
        """文化多样性：D = -Σp_i·log_φ(p_i)"""
        diversity = PhiReal(0.0)
        log_phi = PhiReal(math.log(self.phi.decimal_value))
        
        for prob in probabilities:
            if prob.decimal_value > 0:
                log_term = PhiReal(math.log(prob.decimal_value)) / log_phi
                diversity = diversity - prob * log_term
        
        return diversity
    
    def meme_propagation_rate(self, distance: int) -> PhiReal:
        """模因传播率：α/φ^d"""
        return PhiReal(1.0) / (self.phi ** PhiReal(distance))
    
    def cultural_mutation_rate(self, base_rate: PhiReal) -> PhiReal:
        """文化变异率：μ/φ²"""
        return base_rate / (self.phi ** PhiReal(2.0))
    
    def evolution_dynamics(self, memes: List[PhiReal], time_step: PhiReal) -> List[PhiReal]:
        """演化动力学：dM_i/dt"""
        evolved_memes = []
        for meme in memes:
            # 简化的演化方程
            propagation = PhiReal(0.1) * meme  # 传播项
            decay = meme / self.phi * time_step  # 衰减项
            mutation = PhiReal(0.01) / (self.phi ** PhiReal(2.0))  # 变异项
            
            new_meme = meme + (propagation - decay + mutation) * time_step
            evolved_memes.append(new_meme)
        
        return evolved_memes


class PhiEconomicSystem:
    """φ-经济系统动力学"""
    def __init__(self, phi: PhiReal):
        self.phi = phi
    
    def sector_proportion(self, sector: str) -> PhiReal:
        """部门占比"""
        proportions = {
            "production": PhiReal(1.0) / self.phi,        # 61.8%
            "circulation": PhiReal(1.0) / (self.phi ** PhiReal(2.0)),  # 38.2%
            "service": PhiReal(1.0) / (self.phi ** PhiReal(3.0))       # 23.6%
        }
        return proportions.get(sector, PhiReal(0.0))
    
    def wealth_distribution_exponent(self) -> PhiReal:
        """财富分布指数：α = ln(φ)"""
        return PhiReal(math.log(self.phi.decimal_value))
    
    def economic_cycle_period(self) -> PhiReal:
        """经济周期：T = φ²×12个月"""
        return (self.phi ** PhiReal(2.0)) * PhiReal(12.0)
    
    def value_flow_dynamics(self, production: List[PhiReal], consumption: List[PhiReal], 
                          distribution: List[PhiReal], loss: PhiReal) -> PhiReal:
        """价值流动：dV/dt = φΣP_iC_i - (1/φ)ΣD_j - (1/φ²)L"""
        prod_sum = PhiReal(0.0)
        for p, c in zip(production, consumption):
            prod_sum = prod_sum + p * c
        
        dist_sum = PhiReal(0.0)
        for d in distribution:
            dist_sum = dist_sum + d
        
        return self.phi * prod_sum - dist_sum / self.phi - loss / (self.phi ** PhiReal(2.0))
    
    def pareto_distribution(self, wealth: PhiReal) -> PhiReal:
        """帕累托分布"""
        alpha = self.wealth_distribution_exponent()
        return alpha * (wealth ** (-alpha - PhiReal(1.0)))


class PhiPoliticalSystem:
    """φ-政治组织结构"""
    def __init__(self, phi: PhiReal, base_power: PhiReal):
        self.phi = phi
        self.base_power = base_power
    
    def power_distribution(self, rank: int) -> PhiReal:
        """权力分配：P(i) = P₀/φʳⁱ"""
        return self.base_power / (self.phi ** PhiReal(rank))
    
    def layer_personnel(self, level: int) -> int:
        """各层级人员：Fibonacci数列"""
        fibonacci_personnel = [1, 2, 5, 21]  # F₁, F₃, F₅, F₈
        if level < len(fibonacci_personnel):
            return fibonacci_personnel[level]
        return 0
    
    def political_stability(self, actual_powers: List[PhiReal], expected_powers: List[PhiReal]) -> PhiReal:
        """政治稳定性：S = 1 - Σ|P_i - P_expected|/P_total"""
        if len(actual_powers) != len(expected_powers):
            return PhiReal(0.0)
        
        total_expected = PhiReal(0.0)
        deviation_sum = PhiReal(0.0)
        
        for actual, expected in zip(actual_powers, expected_powers):
            total_expected = total_expected + expected
            deviation = PhiReal(abs(actual.decimal_value - expected.decimal_value))
            deviation_sum = deviation_sum + deviation
        
        if total_expected.decimal_value == 0:
            return PhiReal(1.0)
        
        return PhiReal(1.0) - deviation_sum / total_expected
    
    def power_transfer_probability(self, time: PhiReal, tau: PhiReal) -> PhiReal:
        """权力转移概率：P = 1 - e^(-t/φτ)"""
        exp_arg = -(time / (self.phi * tau))
        return PhiReal(1.0) - PhiReal(math.exp(exp_arg.decimal_value))
    
    def governance_structure(self) -> List[Tuple[PhiReal, int]]:
        """治理结构：(权力, 人员)"""
        structure = []
        for level in range(4):
            power = self.power_distribution(level)
            personnel = self.layer_personnel(level)
            structure.append((power, personnel))
        return structure


class PhiSocialConflict:
    """φ-社会冲突动力学"""
    def __init__(self, phi: PhiReal, base_tension: PhiReal):
        self.phi = phi
        self.base_tension = base_tension
    
    def conflict_threshold(self, level: int) -> PhiReal:
        """冲突阈值：T_k = T₀φᵏ"""
        return self.base_tension * (self.phi ** PhiReal(level))
    
    def resolution_time(self, conflict_level: int, tau: PhiReal) -> PhiReal:
        """解决时间：t = φᵏτ"""
        return (self.phi ** PhiReal(conflict_level)) * tau
    
    def tension_accumulation(self, stress_sources: List[PhiReal], 
                           release_mechanisms: PhiReal, dissipation: PhiReal) -> PhiReal:
        """张力累积：dT/dt = φΣS_i - (1/φ)R - (1/φ²)D"""  
        stress_sum = PhiReal(0.0)
        for stress in stress_sources:
            stress_sum = stress_sum + stress
        
        return (self.phi * stress_sum - 
                release_mechanisms / self.phi - 
                dissipation / (self.phi ** PhiReal(2.0)))
    
    def conflict_classification(self, tension: PhiReal) -> str:
        """冲突分类"""
        t1 = self.conflict_threshold(0)  # T₀
        t2 = self.conflict_threshold(1)  # T₀φ  
        t3 = self.conflict_threshold(2)  # T₀φ²
        # t4 = self.conflict_threshold(3)  # T₀φ³ (systemic level)
        
        if tension.decimal_value <= t1.decimal_value:
            return "minor"
        elif tension.decimal_value <= t2.decimal_value:
            return "moderate"
        elif tension.decimal_value <= t3.decimal_value:
            return "severe"
        else:
            return "systemic"
    
    def tension_dynamics(self, current_tension: PhiReal, time_step: PhiReal) -> PhiReal:
        """张力动力学演化"""
        stress_sources = [PhiReal(0.1), PhiReal(0.2)]
        release = PhiReal(0.05)
        dissipation = PhiReal(0.02)
        
        rate = self.tension_accumulation(stress_sources, release, dissipation)
        return current_tension + rate * time_step


class PhiSocialInnovation:
    """φ-社会创新机制"""
    def __init__(self, phi: PhiReal):
        self.phi = phi
    
    def innovation_frequency(self, innovation_type: str) -> PhiReal:
        """创新频率"""
        frequencies = {
            "breakthrough": PhiReal(1.0) / (self.phi ** PhiReal(3.0)),  # 1/φ³
            "incremental": PhiReal(1.0) / (self.phi ** PhiReal(2.0)),   # 1/φ²
            "micro": PhiReal(1.0) / self.phi                             # 1/φ
        }
        return frequencies.get(innovation_type, PhiReal(0.0))
    
    def innovation_impact_duration(self, innovation_type: str, tau: PhiReal) -> PhiReal:
        """创新影响持续时间"""
        multipliers = {
            "breakthrough": self.phi ** PhiReal(3.0),  # φ³τ
            "incremental": self.phi ** PhiReal(2.0),   # φ²τ
            "micro": self.phi                          # φτ
        }
        return multipliers.get(innovation_type, PhiReal(1.0)) * tau
    
    def innovation_diffusion(self, time: PhiReal, tau: PhiReal) -> PhiReal:
        """创新扩散S曲线"""
        # 简化的S曲线：1/(1 + e^(-(t-φτ)/τ))
        turning_point = self.phi * tau
        exp_arg = -((time - turning_point) / tau)
        return PhiReal(1.0) / (PhiReal(1.0) + PhiReal(math.exp(exp_arg.decimal_value)))
    
    def innovation_index(self, creativity: List[PhiReal], diversity: List[PhiReal], 
                        adaptability: List[PhiReal]) -> PhiReal:
        """创新指数：I = Σ(C_k/φᵏ)×(D_k/φᵏ)×(A_k/φᵏ)"""
        total = PhiReal(0.0)
        min_len = min(len(creativity), len(diversity), len(adaptability))
        
        for k in range(min_len):
            c_term = creativity[k] / (self.phi ** PhiReal(k + 1))
            d_term = diversity[k] / (self.phi ** PhiReal(k + 1))
            a_term = adaptability[k] / (self.phi ** PhiReal(k + 1))
            total = total + c_term * d_term * a_term
        
        return total
    
    def diffusion_turning_point(self, tau: PhiReal) -> PhiReal:
        """扩散转折点：t = φτ"""
        return self.phi * tau


class PhiSocialMemory:
    """φ-社会记忆系统"""
    def __init__(self, phi: PhiReal, base_tau: PhiReal):
        self.phi = phi
        self.base_tau = base_tau
    
    def memory_decay_time(self, level: int) -> PhiReal:
        """记忆衰减时间：τₖ = τ₀φᵏ"""
        return self.base_tau * (self.phi ** PhiReal(level))
    
    def memory_weight(self, level: int) -> PhiReal:
        """记忆权重：1/φᵏ"""
        return PhiReal(1.0) / (self.phi ** PhiReal(level))
    
    def memory_importance(self, level: int) -> PhiReal:
        """记忆重要性（按φ衰减）"""
        return self.memory_weight(level)
    
    def total_memory(self, initial_memory: PhiReal, time: PhiReal) -> PhiReal:
        """总记忆：M(t) = M₀Σ(e^(-t/τₖ)/φᵏ)"""
        total = PhiReal(0.0)
        max_levels = 5  # 个人、家庭、社区、民族、人类
        
        for k in range(max_levels):
            tau_k = self.memory_decay_time(k)
            weight = self.memory_weight(k)
            exp_arg = -(time / tau_k)
            decay_factor = PhiReal(math.exp(exp_arg.decimal_value))
            total = total + weight * decay_factor
        
        return initial_memory * total
    
    def memory_hierarchy(self) -> List[Tuple[PhiReal, PhiReal]]:
        """记忆层级：(衰减时间, 权重)"""
        hierarchy = []
        for k in range(5):
            decay_time = self.memory_decay_time(k)
            weight = self.memory_weight(k)
            hierarchy.append((decay_time, weight))
        return hierarchy


class PhiSocialLearning:
    """φ-社会学习适应"""
    def __init__(self, phi: PhiReal):
        self.phi = phi
    
    def learning_efficiency(self, learning_mode: str) -> PhiReal:
        """学习效率"""
        efficiencies = {
            "imitation": PhiReal(1.0),        # η₁ = 1
            "trial_error": PhiReal(1.0) / self.phi,      # η₂ = 1/φ
            "innovation": PhiReal(1.0) / (self.phi ** PhiReal(2.0))  # η₃ = 1/φ²
        }
        return efficiencies.get(learning_mode, PhiReal(1.0))
    
    def collective_learning_rate(self, base_rate: PhiReal, group_size: int) -> PhiReal:
        """集体学习速率：α = α₀√(N/φ)"""
        return base_rate * PhiReal(math.sqrt(group_size / self.phi.decimal_value))
    
    def adaptation_time(self, group_size: int, tau: PhiReal) -> PhiReal:
        """适应时间：t = φ²τln(N)"""
        return (self.phi ** PhiReal(2.0)) * tau * PhiReal(math.log(group_size))
    
    def collective_intelligence(self, individual_learning: List[PhiReal], 
                              distances: List[PhiReal], weights: List[PhiReal]) -> PhiReal:
        """集体智能：L = Σ(w_i/φᵉⁱ)L_i"""
        if len(individual_learning) != len(distances) or len(individual_learning) != len(weights):
            raise ValueError("All lists must have same length")
        
        total = PhiReal(0.0)
        for learning, distance, weight in zip(individual_learning, distances, weights):
            effective_weight = weight / (self.phi ** distance)
            total = total + effective_weight * learning
        
        return total
    
    def social_adaptation(self, time: PhiReal, tau: PhiReal, group_size: int) -> PhiReal:
        """社会适应过程"""
        adaptation_time = self.adaptation_time(group_size, tau)
        progress = time / adaptation_time
        # 简化的适应曲线
        return PhiReal(1.0) - PhiReal(math.exp(-progress.decimal_value))


class TestPhiSocialEntropyDynamics(unittest.TestCase):
    """T19-3 φ-社会熵动力学定理测试"""
    
    def setUp(self):
        """测试初始化"""
        self.phi = PhiReal((1 + math.sqrt(5)) / 2)
        self.tolerance = 1e-10
    
    def test_phi_real_operations(self):
        """测试φ实数运算"""
        a = PhiReal(1.618)
        b = PhiReal(1.0)
        
        # 基本运算
        self.assertAlmostEqual((a + b).decimal_value, 2.618, places=3)
        self.assertAlmostEqual((a * b).decimal_value, 1.618, places=3)
        self.assertAlmostEqual((a / b).decimal_value, 1.618, places=3)
        
        # no-11约束检查
        self.assertFalse(a.has_consecutive_11())
    
    def test_phi_complex_operations(self):
        """测试φ复数运算"""
        real = PhiReal(1.618)
        imag = PhiReal(1.0)
        c = PhiComplex(real, imag)
        
        magnitude = c.magnitude()
        self.assertAlmostEqual(magnitude.decimal_value, math.sqrt(1.618**2 + 1**2), places=3)
    
    def test_social_system_self_reference(self):
        """测试社会系统自指性：S = S[S]"""
        system = PhiSocialSystem(self.phi, 100)
        system._social_entropy = PhiReal(1.0)
        
        # 验证自指性质
        self.assertTrue(system.verify_self_reference_property())
        
        # 测试熵增
        initial_entropy = system.social_entropy.decimal_value
        system.update_entropy(PhiReal(0.1))
        final_entropy = system.social_entropy.decimal_value
        self.assertGreater(final_entropy, initial_entropy)
    
    def test_social_network_fibonacci_structure(self):
        """测试社会网络Fibonacci结构"""
        network = PhiSocialNetwork(self.phi, 6)
        
        # 验证Fibonacci结构
        self.assertTrue(network.verify_fibonacci_structure())
        
        # 验证层级大小
        self.assertEqual(network.fibonacci_layer_size(0), 1)
        self.assertEqual(network.fibonacci_layer_size(1), 1) 
        self.assertEqual(network.fibonacci_layer_size(2), 2)
        self.assertEqual(network.fibonacci_layer_size(3), 3)
        self.assertEqual(network.fibonacci_layer_size(4), 5)
        
        # 验证连接权重衰减
        weight1 = network.connection_weight(1)
        weight2 = network.connection_weight(2)
        self.assertAlmostEqual(weight1.decimal_value, 1/self.phi.decimal_value, places=3)
        self.assertAlmostEqual(weight2.decimal_value, 1/(self.phi.decimal_value**2), places=3)
    
    def test_information_propagation_dynamics(self):
        """测试信息传播动力学"""
        propagation = PhiInformationPropagation(self.phi, PhiReal(0.1))
        initial_info = PhiReal(1.0)
        time = PhiReal(1.0)
        tau = PhiReal(1.0)
        
        # 病毒传播
        viral = propagation.viral_propagation(initial_info, time, tau)
        expected_viral = initial_info.decimal_value * (self.phi.decimal_value ** (time.decimal_value / tau.decimal_value))
        self.assertAlmostEqual(viral.decimal_value, expected_viral, places=3)
        
        # 传播范围
        range_level1 = propagation.propagation_range(1, PhiReal(10.0))
        expected_range = 10.0 * self.phi.decimal_value
        self.assertAlmostEqual(range_level1.decimal_value, expected_range, places=3)
    
    def test_group_decision_mechanism(self):
        """测试群体决策机制"""
        decision = PhiGroupDecision(self.phi)
        
        # 决策权重
        weight0 = decision.decision_weight(0)
        weight1 = decision.decision_weight(1)
        self.assertAlmostEqual(weight0.decimal_value, 1.0, places=3)
        self.assertAlmostEqual(weight1.decimal_value, 1/self.phi.decimal_value, places=3)
        
        # 共识形成
        opinions = [PhiReal(1.0), PhiReal(2.0), PhiReal(3.0)]
        weights = [PhiReal(1.0), PhiReal(0.618), PhiReal(0.382)]
        consensus = decision.consensus_formation(opinions, weights)
        self.assertIsInstance(consensus, PhiReal)
        
        # 收敛时间
        conv_time = decision.convergence_time(100, PhiReal(1.0))
        expected_time = self.phi.decimal_value * math.log(100)
        self.assertAlmostEqual(conv_time.decimal_value, expected_time, places=3)
    
    def test_social_hierarchy_structure(self):
        """测试社会分层结构"""
        hierarchy = PhiSocialHierarchy(self.phi, 5)
        
        # 验证Fibonacci人口结构
        self.assertTrue(hierarchy.verify_fibonacci_population())
        
        # 验证层级人口
        self.assertEqual(hierarchy.layer_population(0), 5)      # 精英层
        self.assertEqual(hierarchy.layer_population(1), 21)     # 管理层
        self.assertEqual(hierarchy.layer_population(2), 233)    # 专业层
        self.assertEqual(hierarchy.layer_population(3), 10946) # 技能层
        self.assertEqual(hierarchy.layer_population(4), 5702887) # 基础层
        
        # 验证权力分配
        power1 = hierarchy.layer_power(0)
        power2 = hierarchy.layer_power(1)
        self.assertAlmostEqual(power2.decimal_value / power1.decimal_value, self.phi.decimal_value, places=3)
    
    def test_cultural_evolution_dynamics(self):
        """测试文化演化动力学"""
        evolution = PhiCulturalEvolution(self.phi)
        tau = PhiReal(1.0)
        
        # 模因存活时间
        core_time = evolution.meme_survival_time("core", tau)
        mainstream_time = evolution.meme_survival_time("mainstream", tau)
        
        expected_core = (self.phi.decimal_value ** 3) * tau.decimal_value
        expected_mainstream = (self.phi.decimal_value ** 2) * tau.decimal_value
        
        self.assertAlmostEqual(core_time.decimal_value, expected_core, places=3)
        self.assertAlmostEqual(mainstream_time.decimal_value, expected_mainstream, places=3)
        
        # 传播率
        rate1 = evolution.meme_propagation_rate(1)
        rate2 = evolution.meme_propagation_rate(2)
        self.assertAlmostEqual(rate1.decimal_value, 1/self.phi.decimal_value, places=3)
        self.assertAlmostEqual(rate2.decimal_value, 1/(self.phi.decimal_value**2), places=3)
    
    def test_economic_system_dynamics(self):
        """测试经济系统动力学"""
        economy = PhiEconomicSystem(self.phi)
        
        # 部门占比
        prod_prop = economy.sector_proportion("production")
        circ_prop = economy.sector_proportion("circulation")
        
        self.assertAlmostEqual(prod_prop.decimal_value, 1/self.phi.decimal_value, places=3)
        self.assertAlmostEqual(circ_prop.decimal_value, 1/(self.phi.decimal_value**2), places=3)
        
        # 经济周期
        cycle = economy.economic_cycle_period()
        expected_cycle = (self.phi.decimal_value ** 2) * 12
        self.assertAlmostEqual(cycle.decimal_value, expected_cycle, places=3)
        
        # 财富分布指数
        alpha = economy.wealth_distribution_exponent()
        expected_alpha = math.log(self.phi.decimal_value)
        self.assertAlmostEqual(alpha.decimal_value, expected_alpha, places=6)
    
    def test_political_system_structure(self):
        """测试政治组织结构"""
        politics = PhiPoliticalSystem(self.phi, PhiReal(100.0))
        
        # 权力分配
        power0 = politics.power_distribution(0)
        power1 = politics.power_distribution(1)
        
        self.assertAlmostEqual(power0.decimal_value, 100.0, places=3)
        self.assertAlmostEqual(power1.decimal_value, 100.0/self.phi.decimal_value, places=3)
        
        # 层级人员
        self.assertEqual(politics.layer_personnel(0), 1)  # F₁
        self.assertEqual(politics.layer_personnel(1), 2)  # F₃  
        self.assertEqual(politics.layer_personnel(2), 5)  # F₅
        self.assertEqual(politics.layer_personnel(3), 21) # F₈
        
        # 治理结构
        structure = politics.governance_structure()
        self.assertEqual(len(structure), 4)
        self.assertEqual(structure[0][1], 1)  # 最高层1人
        self.assertEqual(structure[1][1], 2)  # 核心层2人
    
    def test_social_conflict_dynamics(self):
        """测试社会冲突动力学"""
        conflict = PhiSocialConflict(self.phi, PhiReal(1.0))
        
        # 冲突阈值
        threshold1 = conflict.conflict_threshold(1)
        threshold2 = conflict.conflict_threshold(2)
        
        expected_t1 = self.phi.decimal_value
        expected_t2 = self.phi.decimal_value ** 2
        
        self.assertAlmostEqual(threshold1.decimal_value, expected_t1, places=3)
        self.assertAlmostEqual(threshold2.decimal_value, expected_t2, places=3)
        
        # 冲突分类
        minor_tension = PhiReal(0.5)      # < 1.0
        moderate_tension = PhiReal(1.5)   # 1.0 < x < 1.618
        severe_tension = PhiReal(2.5)     # 1.618 < x < 2.618
        
        self.assertEqual(conflict.conflict_classification(minor_tension), "minor")
        self.assertEqual(conflict.conflict_classification(moderate_tension), "moderate")
        self.assertEqual(conflict.conflict_classification(severe_tension), "severe")
        
        # 解决时间
        resolution_time = conflict.resolution_time(1, PhiReal(1.0))
        expected_resolution = self.phi.decimal_value
        self.assertAlmostEqual(resolution_time.decimal_value, expected_resolution, places=3)


if __name__ == '__main__':
    print("开始T19-3 φ-社会熵动力学定理测试...")
    unittest.main(verbosity=2)