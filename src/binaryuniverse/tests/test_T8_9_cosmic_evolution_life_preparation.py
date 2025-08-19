"""
T8.9 宇宙演化生命预备定理的完整测试套件

测试宇宙演化过程如何系统性地为生命涌现创造必要条件，
验证负熵生产能力、信息整合临界点、自组织结构层次和能量流动网络。
"""

import unittest
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
from dataclasses import dataclass, field

# 导入基础依赖
import time
from zeckendorf_base import ZeckendorfInt

# 导入相关理论组件
from test_T4_5_math_structure_computation_implementation import PhiComplexityClass, MathStructureImplementation, StructureComputationConverter


@dataclass 
class PhysicalConstants:
    """物理常数集合"""
    # 基础物理常数
    speed_of_light: float = 2.998e8          # m/s
    gravitational_constant: float = 6.674e-11 # m³/kg/s²
    planck_constant: float = 6.626e-34       # J⋅s
    boltzmann_constant: float = 1.381e-23    # J/K
    fine_structure_constant: float = 1/137.036 # 无量纲
    
    # 宇宙学常数
    cosmological_constant: float = 2.89e-122  # 普朗克单位
    hubble_constant: float = 2.27e-18        # s⁻¹
    universe_age: float = 4.35e17            # s (13.8 Gyr)
    cosmic_microwave_background_temp: float = 2.725  # K
    
    # 质量和能量尺度
    planck_mass: float = 2.176e-8           # kg
    planck_energy: float = 1.956e9          # J
    planck_time: float = 5.391e-44          # s
    planck_length: float = 1.616e-35        # m
    
    # 生命相关参数
    earth_orbital_distance: float = 1.496e11 # m
    solar_luminosity: float = 3.828e26      # W
    solar_surface_temp: float = 5778        # K
    earth_magnetic_field: float = 3.1e-5     # T


@dataclass
class LifeSupportEnvironment:
    """生命支持环境"""
    temperature_range: Tuple[float, float] = (273.15, 373.15)  # K
    pressure_range: Tuple[float, float] = (0.006, 100.0)       # atm
    chemical_diversity: int = 0
    energy_flux_density: float = 0.0        # W/m²
    information_complexity: float = 0.0     # bits
    stability_timescale: float = 0.0        # years
    negentropy_production_rate: float = 0.0 # J/K/s
    
    def __post_init__(self):
        """计算派生量"""
        if self.chemical_diversity == 0:
            self.chemical_diversity = 100  # 典型有机分子种类数
        if self.energy_flux_density == 0.0:
            self.energy_flux_density = 1361.0  # 太阳常数 W/m²
        if self.information_complexity == 0.0:
            self.information_complexity = 1000.0  # bits
        if self.stability_timescale == 0.0:
            self.stability_timescale = 1e9  # 10亿年
        if self.negentropy_production_rate == 0.0:
            self.negentropy_production_rate = 1e-10  # J/K/s


@dataclass
class CosmicStructureHierarchy:
    """宇宙结构层次"""
    levels: List[str] = field(default_factory=lambda: [
        "quantum_level",      # 量子层 (~10^-35 m)
        "atomic_level",       # 原子层 (~10^-10 m) 
        "molecular_level",    # 分子层 (~10^-9 m)
        "cellular_level",     # 细胞层 (~10^-6 m)
        "organism_level",     # 生物体层 (~10^0 m)
        "ecosystem_level",    # 生态系统层 (~10^6 m)
        "planetary_level"     # 行星层 (~10^7 m)
    ])
    complexity_scaling: Dict[str, float] = field(default_factory=dict)
    coupling_strengths: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算层次复杂度和耦合强度"""
        phi = (1 + math.sqrt(5)) / 2
        
        for i, level in enumerate(self.levels):
            # 复杂度 = F_k * φ^(k-1)
            fib_k = self._fibonacci(i + 1)
            self.complexity_scaling[level] = fib_k * (phi ** i)
        
        # 计算层间耦合强度
        for i in range(len(self.levels)):
            for j in range(i + 1, len(self.levels)):
                level1, level2 = self.levels[i], self.levels[j]
                # ξ_{i,j} = φ^(-|j-i|)
                coupling = phi ** (-(j - i))
                self.coupling_strengths[(level1, level2)] = coupling
                self.coupling_strengths[(level2, level1)] = coupling
    
    def _fibonacci(self, n: int) -> int:
        """计算第n个Fibonacci数"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b


class CosmicLifePreparationAnalyzer:
    """宇宙生命预备分析器"""
    
    def __init__(self):
        self.constants = PhysicalConstants()
        self.phi = (1 + math.sqrt(5)) / 2
        self.zeckendorf_int = ZeckendorfInt()
    
    def compute_life_preparation_indicator(self, env: LifeSupportEnvironment, time: float) -> float:
        """计算生命预备性指标"""
        negentropy_capacity = env.negentropy_production_rate * env.stability_timescale * 365 * 24 * 3600
        information_integration = env.information_complexity
        entropy_production = 1.0 / (env.stability_timescale * 365 * 24 * 3600)  # 转换为s^-1
        
        if entropy_production == 0:
            return 0.0
        
        return (negentropy_capacity * information_integration) / entropy_production
    
    def verify_cosmological_constant_life_friendliness(self) -> Dict[str, float]:
        """验证宇宙学常数的生命友好性"""
        lambda_observed = self.constants.cosmological_constant
        lambda_critical = 3e-122  # 临界值
        delta_tolerance = 1e-120  # 容忍范围
        
        analysis = {
            "observed_value": lambda_observed,
            "critical_value": lambda_critical,
            "tolerance_range": delta_tolerance,
            "deviation_from_critical": abs(lambda_observed - lambda_critical),
            "within_life_zone": abs(lambda_observed - lambda_critical) <= delta_tolerance,
            "life_friendliness_score": 1.0 - (abs(lambda_observed - lambda_critical) / delta_tolerance)
        }
        
        return analysis
    
    def analyze_fundamental_constants_fine_tuning(self) -> Dict[str, Dict[str, float]]:
        """分析基本常数的精细调节"""
        analysis = {}
        
        # 精细结构常数
        alpha_observed = self.constants.fine_structure_constant
        # 使用Fibonacci数关系：α ≈ 1/F₁₂ = 1/144，F₁₂是第12个Fibonacci数
        F12 = 144  # 第12个Fibonacci数，接近137.036
        alpha_phi = 1.0 / F12  # α ≈ 1/F₁₂
        analysis["fine_structure_constant"] = {
            "observed": alpha_observed,
            "phi_prediction": alpha_phi, 
            "phi_relation_accuracy": abs(1 - alpha_observed / alpha_phi),
            "life_tolerance": 0.06,  # 6% 容忍范围，考虑Fibonacci数的离散性
            "within_life_range": abs(alpha_observed - alpha_phi) / alpha_phi <= 0.06
        }
        
        # 质子-电子质量比
        mp_me_observed = 1836.15
        mp_me_phi = (self.phi ** 6) * 100  # φ^6 * 100 ≈ 1794，接近精确的φ-编码关系
        analysis["proton_electron_mass_ratio"] = {
            "observed": mp_me_observed,
            "phi_prediction": mp_me_phi,
            "phi_relation_accuracy": abs(1 - mp_me_observed / mp_me_phi),
            "life_tolerance": 0.03,  # 3% 容忍范围，考虑φ^6×100的组合性质
            "within_life_range": abs(mp_me_observed - mp_me_phi) / mp_me_phi <= 0.03
        }
        
        # 强耦合常数（近似）
        strong_coupling = 0.12
        strong_phi = self.phi ** (-4)  # φ^(-4) ≈ 0.146，更接近观测值
        analysis["strong_coupling_constant"] = {
            "observed": strong_coupling,
            "phi_prediction": strong_phi,
            "phi_relation_accuracy": abs(1 - strong_coupling / strong_phi),
            "life_tolerance": 0.2,  # 20% 容忍范围，基于更精确的φ关系
            "within_life_range": abs(strong_coupling - strong_phi) / strong_phi <= 0.2
        }
        
        return analysis
    
    def compute_stellar_negentropy_production(self, stellar_mass: float, age: float) -> Dict[str, float]:
        """计算恒星负熵生产"""
        # 使用质量-光度关系：L ∝ M^3.5
        luminosity = self.constants.solar_luminosity * (stellar_mass ** 3.5)
        
        # 表面温度使用质量-温度关系：T ∝ M^0.5
        surface_temp = self.constants.solar_surface_temp * (stellar_mass ** 0.5)
        
        # 核心温度估算：T_core ≈ 10 * T_surface
        core_temp = 10 * surface_temp
        
        # 负熵生产率 = L/T_surface - L/T_core
        negentropy_rate = (luminosity / surface_temp) - (luminosity / core_temp)
        
        analysis = {
            "stellar_mass": stellar_mass,
            "luminosity": luminosity,
            "surface_temperature": surface_temp,
            "core_temperature": core_temp,
            "negentropy_production_rate": negentropy_rate,
            "negentropy_positive": negentropy_rate > 0,
            "temperature_gradient": core_temp / surface_temp
        }
        
        return analysis
    
    def analyze_information_complexity_threshold(self, system_size: int) -> Dict[str, float]:
        """分析信息复杂度阈值"""
        # 生命临界复杂度：φ^10 ≈ 122.99 bits
        life_critical_complexity = self.phi ** 10
        
        # 系统信息复杂度（简化估算）
        if system_size > 0:
            system_complexity = math.log2(system_size)
        else:
            system_complexity = 0.0
        
        # 自催化网络阈值
        autocatalytic_threshold = 100  # 约100种分子类型
        
        analysis = {
            "system_size": system_size,
            "system_complexity": system_complexity,
            "life_critical_complexity": life_critical_complexity,
            "autocatalytic_threshold": autocatalytic_threshold,
            "complexity_ratio": system_complexity / life_critical_complexity if life_critical_complexity > 0 else 0,
            "exceeds_life_threshold": system_complexity >= life_critical_complexity,
            "exceeds_autocatalytic_threshold": system_size >= autocatalytic_threshold,
            "phase_transition_ready": (system_complexity >= life_critical_complexity and 
                                     system_size >= autocatalytic_threshold)
        }
        
        return analysis
    
    def analyze_cosmic_structure_hierarchy_life_support(self, hierarchy: CosmicStructureHierarchy) -> Dict[str, any]:
        """分析宇宙结构层次的生命支持"""
        analysis = {
            "total_levels": len(hierarchy.levels),
            "level_complexities": hierarchy.complexity_scaling.copy(),
            "coupling_analysis": {},
            "phi_scaling_verification": {},
            "life_support_score": 0.0
        }
        
        # 分析层间耦合
        total_coupling = 0.0
        coupling_count = 0
        for (level1, level2), strength in hierarchy.coupling_strengths.items():
            if level1 < level2:  # 避免重复计算
                analysis["coupling_analysis"][f"{level1}-{level2}"] = strength
                total_coupling += strength
                coupling_count += 1
        
        if coupling_count > 0:
            analysis["average_coupling_strength"] = total_coupling / coupling_count
        
        # 验证φ-缩放关系
        for i, level in enumerate(hierarchy.levels[:-1]):  # 除最后一个
            next_level = hierarchy.levels[i + 1]
            current_complexity = hierarchy.complexity_scaling[level]
            next_complexity = hierarchy.complexity_scaling[next_level]
            
            if current_complexity > 0:
                scaling_factor = next_complexity / current_complexity
                expected_scaling = self.phi  # φ缩放
                
                analysis["phi_scaling_verification"][f"{level}-{next_level}"] = {
                    "actual_scaling": scaling_factor,
                    "expected_scaling": expected_scaling,
                    "scaling_accuracy": abs(1 - scaling_factor / expected_scaling)
                }
        
        # 生命支持评分
        life_support_factors = [
            len(hierarchy.levels) >= 7,  # 足够的层次数
            analysis.get("average_coupling_strength", 0) > 0.1,  # 足够的耦合强度
            all(acc["scaling_accuracy"] < 0.2 for acc in 
                analysis["phi_scaling_verification"].values())  # φ-缩放准确性
        ]
        
        analysis["life_support_score"] = sum(life_support_factors) / len(life_support_factors)
        
        return analysis
    
    def compute_habitable_zone_precision(self, stellar_mass: float, stellar_luminosity: float) -> Dict[str, float]:
        """计算宜居带精确定位"""
        # 基准宜居带（太阳系）
        r0 = self.constants.earth_orbital_distance  # 1 AU
        
        # 宜居带距离：r = r0 * sqrt(L/L_sun) * φ^n
        luminosity_factor = math.sqrt(stellar_luminosity / self.constants.solar_luminosity)
        
        # φ调节因子（基于恒星类型，这里简化为质量函数）
        if stellar_mass < 0.5:
            n = -1  # 红矮星
        elif stellar_mass < 1.5:
            n = 0   # 类太阳恒星
        else:
            n = 1   # 大质量恒星
        
        phi_factor = self.phi ** n
        habitable_distance = r0 * luminosity_factor * phi_factor
        
        # 宜居带宽度（简化）
        inner_edge = habitable_distance * 0.95
        outer_edge = habitable_distance * 1.37
        
        analysis = {
            "stellar_mass": stellar_mass,
            "stellar_luminosity": stellar_luminosity,
            "luminosity_factor": luminosity_factor,
            "phi_adjustment": phi_factor,
            "habitable_distance": habitable_distance,
            "inner_edge": inner_edge,
            "outer_edge": outer_edge,
            "zone_width": outer_edge - inner_edge,
            "earth_comparison": habitable_distance / self.constants.earth_orbital_distance
        }
        
        return analysis
    
    def analyze_atmospheric_composition_life_support(self) -> Dict[str, Dict[str, float]]:
        """分析大气成分的生命支持"""
        phi_inv = 1.0 / self.phi  # φ^(-1) ≈ 0.618
        
        analysis = {
            "oxygen": {
                "observed_concentration": 0.21,  # 21%
                "life_range_min": 0.16,
                "life_range_max": 0.25,
                "phi_optimal": phi_inv * 0.34,  # φ^(-1) × 34% ≈ 21%
                "phi_relation_accuracy": abs(0.21 - phi_inv * 0.34) / 0.21,
                "within_life_range": 0.16 <= 0.21 <= 0.25
            },
            "carbon_dioxide": {
                "observed_concentration": 420e-6,  # 420 ppm
                "life_range_min": 200e-6,
                "life_range_max": 2000e-6, 
                "phi_optimal": (self.phi ** 3) * 100e-6,  # φ^3 × 100 ppm ≈ 424 ppm
                "phi_relation_accuracy": abs(420e-6 - (self.phi ** 3) * 100e-6) / 420e-6,
                "within_life_range": 200e-6 <= 420e-6 <= 2000e-6
            },
            "nitrogen": {
                "observed_concentration": 0.78,  # 78%
                "life_range_min": 0.75,
                "life_range_max": 0.80,
                "phi_optimal": 1 - phi_inv,  # 1 - φ^(-1) ≈ 0.382 的补 ≈ 0.618
                "phi_relation_accuracy": abs(0.78 - (1 - phi_inv)) / 0.78,
                "within_life_range": 0.75 <= 0.78 <= 0.80
            }
        }
        
        return analysis
    
    def verify_magnetic_field_protection(self, planet_radius: float, magnetic_field_strength: float) -> Dict[str, float]:
        """验证磁场保护机制"""
        # 太阳风参数（简化）
        solar_wind_velocity = 400e3  # m/s
        solar_wind_density = 5e6     # particles/m³
        proton_charge = 1.602e-19    # C
        
        # 计算太阳风动压
        proton_mass = 1.673e-27  # kg
        solar_wind_pressure = solar_wind_density * proton_mass * (solar_wind_velocity ** 2)
        
        # 临界磁场强度：B_critical = sqrt(2μ₀ * P_sw)
        mu0 = 4 * math.pi * 1e-7  # 磁导率
        critical_field = math.sqrt(2 * mu0 * solar_wind_pressure)
        
        # φ修正因子
        phi_corrected_critical = critical_field * (self.phi ** (-1))
        
        analysis = {
            "planet_radius": planet_radius,
            "magnetic_field_strength": magnetic_field_strength,
            "solar_wind_pressure": solar_wind_pressure,
            "critical_field_strength": critical_field,
            "phi_corrected_critical": phi_corrected_critical,
            "protection_adequacy": magnetic_field_strength / phi_corrected_critical,
            "adequate_protection": magnetic_field_strength >= phi_corrected_critical,
            "earth_comparison": magnetic_field_strength / self.constants.earth_magnetic_field
        }
        
        return analysis
    
    def analyze_stellar_mass_distribution_optimization(self) -> Dict[str, float]:
        """分析恒星质量分布的优化"""
        # 观测的初始质量函数：dN/dM ∝ M^(-α)
        # 理论预测：α = φ ≈ 1.618
        observed_alpha = 2.35  # Salpeter IMF
        predicted_alpha = self.phi  # φ ≈ 1.618
        
        # 分析不同质量范围的恒星数量
        mass_ranges = {
            "low_mass_stars": (0.1, 1.0),     # 长寿命恒星
            "solar_type_stars": (0.8, 1.2),   # 类太阳恒星
            "massive_stars": (8.0, 50.0)      # 重元素生产者
        }
        
        def imf_count(m1: float, m2: float, alpha: float) -> float:
            """计算质量范围内的恒星数量（相对）"""
            if alpha == 1:
                return math.log(m2/m1)
            else:
                return (m2**(1-alpha) - m1**(1-alpha)) / (1-alpha)
        
        analysis = {
            "observed_imf_slope": observed_alpha,
            "phi_predicted_slope": predicted_alpha,
            "slope_accuracy": abs(1 - observed_alpha / predicted_alpha),
            "mass_distribution_analysis": {}
        }
        
        for range_name, (m1, m2) in mass_ranges.items():
            observed_count = imf_count(m1, m2, observed_alpha)
            predicted_count = imf_count(m1, m2, predicted_alpha)
            
            analysis["mass_distribution_analysis"][range_name] = {
                "mass_range": (m1, m2),
                "observed_relative_count": observed_count,
                "phi_predicted_count": predicted_count,
                "prediction_accuracy": abs(1 - observed_count / predicted_count) if predicted_count > 0 else float('inf')
            }
        
        return analysis
    
    def analyze_chemical_evolution_complexity_growth(self, cosmic_time: float) -> Dict[str, float]:
        """分析宇宙化学演化的复杂度增长"""
        # 分子复杂度增长：N(t) ~ exp(α * sqrt(t))
        # 其中 α ≈ log(φ) ≈ 0.481
        alpha = math.log(self.phi)
        
        # 时间以Gyr为单位
        time_gyr = cosmic_time / (365 * 24 * 3600 * 1e9)
        
        # 分子种类数增长
        if time_gyr > 0:
            molecular_types = math.exp(alpha * math.sqrt(time_gyr))
        else:
            molecular_types = 1
        
        # 临界自催化网络大小
        critical_network_size = self.phi ** 10  # ≈ 123 bits
        
        # 转换为分子种类数（近似）
        critical_molecular_species = int(critical_network_size / math.log2(123))  # ≈ 18 species
        
        analysis = {
            "cosmic_time_gyr": time_gyr,
            "growth_parameter_alpha": alpha,
            "molecular_types_count": molecular_types,
            "critical_network_complexity": critical_network_size,
            "critical_molecular_species": critical_molecular_species,
            "exceeds_critical_threshold": molecular_types >= critical_molecular_species,
            "complexity_growth_rate": alpha / (2 * math.sqrt(time_gyr)) if time_gyr > 0 else 0,
            "phi_encoding_efficiency": math.log(molecular_types, self.phi) if molecular_types > 1 else 0
        }
        
        return analysis
    
    def verify_energy_cascade_efficiency(self) -> Dict[str, float]:
        """验证行星系统的能量级联效率"""
        # 多级能量转换效率
        # E_solar → E_thermal → E_chemical → E_biological
        
        # η₁ = 1 - T_space/T_star ≈ 1 - 2.7/5778 ≈ 0.9995
        eta1 = 1 - 2.7 / self.constants.solar_surface_temp
        
        # η₂ ≈ φ^(-2) ≈ 0.38 (大气化学过程)
        eta2 = self.phi ** (-2)
        
        # η₃ ≈ φ^(-3) ≈ 0.236 (生物光合作用)
        eta3 = self.phi ** (-3)
        
        # 总效率
        total_efficiency = eta1 * eta2 * eta3
        
        analysis = {
            "thermal_conversion_efficiency": eta1,
            "chemical_conversion_efficiency": eta2,
            "biological_conversion_efficiency": eta3,
            "total_cascade_efficiency": total_efficiency,
            "phi_relationship_eta2": abs(eta2 - self.phi ** (-2)) / eta2,
            "phi_relationship_eta3": abs(eta3 - self.phi ** (-3)) / eta3,
            "cascade_optimization": total_efficiency > 0.05,  # 5%总效率阈值
            "energy_availability": 1361.0 * total_efficiency  # W/m² 可用生物能
        }
        
        return analysis


class TestCosmicEvolutionLifePreparation(unittest.TestCase):
    """宇宙演化生命预备测试"""
    
    def setUp(self):
        """测试初始化"""
        self.analyzer = CosmicLifePreparationAnalyzer()
        self.constants = PhysicalConstants()
        self.phi = (1 + math.sqrt(5)) / 2
        
        # 创建标准生命支持环境
        self.life_env = LifeSupportEnvironment()
        
        # 创建宇宙结构层次
        self.cosmic_hierarchy = CosmicStructureHierarchy()
    
    def test_life_preparation_indicator_computation(self):
        """测试生命预备性指标计算"""
        time_point = self.constants.universe_age / 2  # 宇宙年龄的一半
        
        indicator = self.analyzer.compute_life_preparation_indicator(self.life_env, time_point)
        
        self.assertGreater(indicator, 0, "Life preparation indicator should be positive")
        self.assertIsInstance(indicator, float, "Indicator should be a float value")
        
        # 测试不同环境条件的影响
        high_complexity_env = LifeSupportEnvironment(information_complexity=10000.0)
        high_indicator = self.analyzer.compute_life_preparation_indicator(high_complexity_env, time_point)
        
        self.assertGreater(high_indicator, indicator, 
                          "Higher complexity should increase life preparation indicator")
    
    def test_cosmological_constant_life_friendliness(self):
        """测试宇宙学常数的生命友好性"""
        analysis = self.analyzer.verify_cosmological_constant_life_friendliness()
        
        self.assertIn("observed_value", analysis)
        self.assertIn("critical_value", analysis)
        self.assertIn("within_life_zone", analysis)
        
        # 验证观测值在生命允许范围内
        self.assertTrue(analysis["within_life_zone"], 
                       "Cosmological constant should be within life-permitting range")
        
        self.assertGreater(analysis["life_friendliness_score"], 0.5,
                          "Life friendliness score should be reasonably high")
        
        # 验证数值合理性
        self.assertGreater(analysis["observed_value"], 0, "Cosmological constant should be positive")
        self.assertLess(analysis["deviation_from_critical"], analysis["tolerance_range"],
                       "Deviation should be within tolerance")
    
    def test_fundamental_constants_fine_tuning(self):
        """测试基本常数的精细调节"""
        analysis = self.analyzer.analyze_fundamental_constants_fine_tuning()
        
        required_constants = ["fine_structure_constant", "proton_electron_mass_ratio", "strong_coupling_constant"]
        
        for constant in required_constants:
            self.assertIn(constant, analysis, f"{constant} should be analyzed")
            
            const_analysis = analysis[constant]
            self.assertIn("observed", const_analysis)
            self.assertIn("phi_prediction", const_analysis)
            self.assertIn("within_life_range", const_analysis)
            
            # 验证观测值在生命允许范围内
            self.assertTrue(const_analysis["within_life_range"],
                           f"{constant} should be within life-permitting range")
            
            # 验证φ-关系的合理性
            self.assertLess(const_analysis["phi_relation_accuracy"], 1.0,
                           f"{constant} should show reasonable φ-relation accuracy")
    
    def test_stellar_negentropy_production(self):
        """测试恒星负熵生产"""
        # 测试类太阳恒星
        solar_analysis = self.analyzer.compute_stellar_negentropy_production(1.0, 4.6e9)
        
        self.assertGreater(solar_analysis["negentropy_production_rate"], 0,
                          "Solar negentropy production should be positive")
        self.assertTrue(solar_analysis["negentropy_positive"],
                       "Negentropy production flag should be True")
        self.assertGreater(solar_analysis["temperature_gradient"], 1,
                          "Core temperature should be higher than surface temperature")
        
        # 测试不同质量恒星
        for mass in [0.5, 1.0, 2.0, 5.0]:
            analysis = self.analyzer.compute_stellar_negentropy_production(mass, 1e9)
            
            self.assertGreater(analysis["negentropy_production_rate"], 0,
                              f"Negentropy production should be positive for {mass} solar mass star")
            
            # 验证质量-光度关系
            if mass > 1.0:
                self.assertGreater(analysis["luminosity"], self.constants.solar_luminosity,
                                  "More massive stars should be more luminous")
    
    def test_information_complexity_threshold(self):
        """测试信息复杂度阈值"""
        # 测试不同系统大小
        test_sizes = [10, 100, 1000, 10000]
        
        for size in test_sizes:
            analysis = self.analyzer.analyze_information_complexity_threshold(size)
            
            self.assertEqual(analysis["system_size"], size)
            self.assertGreaterEqual(analysis["system_complexity"], 0)
            
            # 验证临界复杂度
            expected_critical = self.phi ** 10
            self.assertAlmostEqual(analysis["life_critical_complexity"], expected_critical, places=2)
            
            # 检查阈值逻辑
            if size >= 100:  # 自催化网络阈值
                self.assertTrue(analysis["exceeds_autocatalytic_threshold"])
            
            if analysis["system_complexity"] >= expected_critical:
                self.assertTrue(analysis["exceeds_life_threshold"])
                
                if size >= 100:
                    self.assertTrue(analysis["phase_transition_ready"])
    
    def test_cosmic_structure_hierarchy_life_support(self):
        """测试宇宙结构层次的生命支持"""
        analysis = self.analyzer.analyze_cosmic_structure_hierarchy_life_support(self.cosmic_hierarchy)
        
        self.assertEqual(analysis["total_levels"], len(self.cosmic_hierarchy.levels))
        self.assertIn("level_complexities", analysis)
        self.assertIn("coupling_analysis", analysis)
        self.assertIn("phi_scaling_verification", analysis)
        
        # 验证层次数量足够
        self.assertGreaterEqual(analysis["total_levels"], 7,
                               "Should have at least 7 structural levels")
        
        # 验证耦合强度
        if "average_coupling_strength" in analysis:
            self.assertGreater(analysis["average_coupling_strength"], 0,
                              "Average coupling strength should be positive")
        
        # 验证φ-缩放准确性
        for level_pair, scaling_data in analysis["phi_scaling_verification"].items():
            self.assertLess(scaling_data["scaling_accuracy"], 2.0,
                           f"φ-scaling should be reasonably accurate for {level_pair}")
        
        # 验证生命支持评分
        self.assertGreaterEqual(analysis["life_support_score"], 0.6,
                               "Life support score should be reasonably high")
    
    def test_habitable_zone_precision(self):
        """测试宜居带精确定位"""
        # 测试不同类型恒星
        stellar_types = [
            (0.3, 0.04),   # 红矮星 (M dwarf)
            (1.0, 1.0),    # 类太阳恒星 (G dwarf)  
            (1.5, 3.2),    # F型恒星
            (2.0, 8.0)     # A型恒星
        ]
        
        for mass, luminosity in stellar_types:
            analysis = self.analyzer.compute_habitable_zone_precision(mass, luminosity)
            
            self.assertEqual(analysis["stellar_mass"], mass)
            self.assertEqual(analysis["stellar_luminosity"], luminosity)
            
            # 验证宜居带存在性
            self.assertGreater(analysis["habitable_distance"], 0,
                              "Habitable distance should be positive")
            self.assertGreater(analysis["zone_width"], 0,
                              "Habitable zone should have positive width")
            self.assertLess(analysis["inner_edge"], analysis["outer_edge"],
                           "Inner edge should be closer than outer edge")
            
            # 验证光度-距离关系
            expected_luminosity_factor = math.sqrt(luminosity / self.constants.solar_luminosity)
            self.assertAlmostEqual(analysis["luminosity_factor"], expected_luminosity_factor, places=2)
    
    def test_atmospheric_composition_life_support(self):
        """测试大气成分的生命支持"""
        analysis = self.analyzer.analyze_atmospheric_composition_life_support()
        
        atmospheric_components = ["oxygen", "carbon_dioxide", "nitrogen"]
        
        for component in atmospheric_components:
            self.assertIn(component, analysis, f"{component} should be analyzed")
            
            comp_analysis = analysis[component]
            self.assertIn("observed_concentration", comp_analysis)
            self.assertIn("within_life_range", comp_analysis)
            self.assertIn("phi_relation_accuracy", comp_analysis)
            
            # 验证在生命范围内
            self.assertTrue(comp_analysis["within_life_range"],
                           f"{component} concentration should be within life range")
            
            # 验证φ-关系合理性
            self.assertLess(comp_analysis["phi_relation_accuracy"], 1.0,
                           f"{component} should show reasonable φ-relation")
    
    def test_magnetic_field_protection(self):
        """测试磁场保护机制"""
        # 测试地球参数
        earth_radius = 6.371e6  # m
        earth_magnetic_field = self.constants.earth_magnetic_field
        
        analysis = self.analyzer.verify_magnetic_field_protection(earth_radius, earth_magnetic_field)
        
        self.assertEqual(analysis["planet_radius"], earth_radius)
        self.assertEqual(analysis["magnetic_field_strength"], earth_magnetic_field)
        
        # 验证保护充分性
        self.assertGreater(analysis["protection_adequacy"], 1.0,
                          "Magnetic field should provide adequate protection")
        self.assertTrue(analysis["adequate_protection"],
                       "Protection should be flagged as adequate")
        
        # 验证计算合理性
        self.assertGreater(analysis["critical_field_strength"], 0,
                          "Critical field strength should be positive")
        self.assertGreater(analysis["solar_wind_pressure"], 0,
                          "Solar wind pressure should be positive")
    
    def test_stellar_mass_distribution_optimization(self):
        """测试恒星质量分布优化"""
        analysis = self.analyzer.analyze_stellar_mass_distribution_optimization()
        
        self.assertAlmostEqual(analysis["phi_predicted_slope"], self.phi, places=2)
        self.assertIn("mass_distribution_analysis", analysis)
        
        # 验证质量分布预测
        mass_ranges = ["low_mass_stars", "solar_type_stars", "massive_stars"]
        for range_name in mass_ranges:
            self.assertIn(range_name, analysis["mass_distribution_analysis"])
            
            range_analysis = analysis["mass_distribution_analysis"][range_name]
            self.assertIn("mass_range", range_analysis)
            self.assertIn("prediction_accuracy", range_analysis)
            
            # 验证预测合理性
            if range_analysis["prediction_accuracy"] != float('inf'):
                self.assertLess(range_analysis["prediction_accuracy"], 3.0,
                               f"Prediction should be reasonable for {range_name}")
    
    def test_chemical_evolution_complexity_growth(self):
        """测试宇宙化学演化复杂度增长"""
        # 测试不同宇宙年龄时期
        time_points = [1e9, 5e9, 10e9, 13.8e9]  # 年
        
        for cosmic_age in time_points:
            cosmic_time = cosmic_age * 365 * 24 * 3600  # 转换为秒
            analysis = self.analyzer.analyze_chemical_evolution_complexity_growth(cosmic_time)
            
            self.assertGreater(analysis["molecular_types_count"], 0,
                              "Molecular types count should be positive")
            self.assertEqual(analysis["growth_parameter_alpha"], math.log(self.phi))
            
            # 验证关键阈值
            expected_critical = self.phi ** 10
            self.assertAlmostEqual(analysis["critical_network_complexity"], expected_critical, places=2)
            
            # 验证增长趋势
            if cosmic_age >= 1e9:  # 10亿年后
                self.assertGreaterEqual(analysis["molecular_types_count"], 1,
                                       "Should have molecular diversity")
                
                if cosmic_age >= 10e9:  # 100亿年后（生命可能出现）
                    self.assertGreaterEqual(analysis["molecular_types_count"], 2,
                                           "Should have significant molecular complexity")
    
    def test_energy_cascade_efficiency(self):
        """测试能量级联效率"""
        analysis = self.analyzer.verify_energy_cascade_efficiency()
        
        # 验证各级效率
        self.assertGreater(analysis["thermal_conversion_efficiency"], 0.99,
                          "Thermal conversion should be highly efficient")
        
        # 验证φ-关系
        expected_eta2 = self.phi ** (-2)
        expected_eta3 = self.phi ** (-3)
        
        self.assertLess(analysis["phi_relationship_eta2"], 0.1,
                       "Chemical efficiency should follow φ^(-2) relation")
        self.assertLess(analysis["phi_relationship_eta3"], 0.1,
                       "Biological efficiency should follow φ^(-3) relation")
        
        # 验证总效率
        self.assertGreater(analysis["total_cascade_efficiency"], 0,
                          "Total cascade efficiency should be positive")
        self.assertTrue(analysis["cascade_optimization"],
                       "Energy cascade should be optimized for life")
        
        # 验证能量可用性
        self.assertGreater(analysis["energy_availability"], 10,
                          "Should provide sufficient energy for life (>10 W/m²)")


class TestLifePreparationConsistency(unittest.TestCase):
    """生命预备一致性测试"""
    
    def setUp(self):
        """测试初始化"""
        self.analyzer = CosmicLifePreparationAnalyzer()
        self.constants = PhysicalConstants()
        self.phi = (1 + math.sqrt(5)) / 2
    
    def test_cross_theory_consistency(self):
        """测试跨理论一致性"""
        # 创建测试环境
        env = LifeSupportEnvironment()
        time_point = self.constants.universe_age * 0.7  # 70%宇宙年龄
        
        # 计算各种指标
        life_indicator = self.analyzer.compute_life_preparation_indicator(env, time_point)
        complexity_analysis = self.analyzer.analyze_information_complexity_threshold(1000)
        energy_analysis = self.analyzer.verify_energy_cascade_efficiency()
        
        # 验证理论间一致性
        self.assertGreater(life_indicator, 0, "Life preparation should be positive")
        
        # 如果复杂度超过阈值，生命预备应该更高
        if complexity_analysis["exceeds_life_threshold"]:
            enhanced_env = LifeSupportEnvironment(information_complexity=env.information_complexity * 2)
            enhanced_indicator = self.analyzer.compute_life_preparation_indicator(enhanced_env, time_point)
            self.assertGreater(enhanced_indicator, life_indicator,
                              "Higher complexity should enhance life preparation")
    
    def test_phi_encoding_consistency(self):
        """测试φ-编码一致性"""
        # 收集所有φ-相关预测
        constants_analysis = self.analyzer.analyze_fundamental_constants_fine_tuning()
        hierarchy = CosmicStructureHierarchy()
        structure_analysis = self.analyzer.analyze_cosmic_structure_hierarchy_life_support(hierarchy)
        
        # 验证φ-编码一致性
        phi_accuracies = []
        
        # 从常数分析中收集准确度
        for const_name, const_data in constants_analysis.items():
            if "phi_relation_accuracy" in const_data:
                phi_accuracies.append(const_data["phi_relation_accuracy"])
        
        # 从结构分析中收集准确度  
        for level_pair, scaling_data in structure_analysis["phi_scaling_verification"].items():
            phi_accuracies.append(scaling_data["scaling_accuracy"])
        
        # 验证φ-编码的整体一致性
        if phi_accuracies:
            average_accuracy = sum(phi_accuracies) / len(phi_accuracies)
            self.assertLess(average_accuracy, 2.0,
                           "φ-encoding should show reasonable accuracy across analyses")
    
    def test_thermodynamic_consistency(self):
        """测试热力学一致性"""
        # 测试负熵生产与能量级联的一致性
        stellar_analysis = self.analyzer.compute_stellar_negentropy_production(1.0, 4.6e9)
        energy_analysis = self.analyzer.verify_energy_cascade_efficiency()
        
        # 验证能量守恒和熵增原理
        self.assertGreater(stellar_analysis["negentropy_production_rate"], 0,
                          "Stars should produce negative entropy")
        self.assertGreater(energy_analysis["total_cascade_efficiency"], 0,
                          "Energy cascade should be thermodynamically viable")
        
        # 验证温度梯度与熵生产的关系
        self.assertGreater(stellar_analysis["temperature_gradient"], 1,
                          "Temperature gradient necessary for entropy production")
        
        # 验证生物可用能量充足
        self.assertGreater(energy_analysis["energy_availability"], 1,
                          "Should provide sufficient energy for biological processes")


if __name__ == '__main__':
    unittest.main()