#!/usr/bin/env python3
"""
Classification Statistics Generator v3.0
生成T{n}理论五类分类系统的详细统计报告
"""

import math
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path

try:
    from .theory_parser import TheoryParser, FibonacciOperationType
    from .theory_validator import PrimeChecker
except ImportError:
    from theory_parser import TheoryParser, FibonacciOperationType
    from theory_validator import PrimeChecker

class ClassificationStatistics:
    """五类理论分类统计分析器"""
    
    def __init__(self, max_theory: int = 997):
        self.max_theory = max_theory
        self.parser = TheoryParser(max_theory)
        self.prime_checker = PrimeChecker()
        self.phi = (1 + math.sqrt(5)) / 2  # 黄金比例
        
        # 生成基础数据
        self.fibonacci_set = set(self.parser.fibonacci_sequence)
        self.primes = self.prime_checker.get_primes_up_to(max_theory)
        self.prime_set = set(self.primes)
        
    def classify_theory(self, n: int) -> str:
        """对理论进行五类分类"""
        if n == 1:
            return "AXIOM"
        elif n in self.fibonacci_set and n in self.prime_set:
            return "PRIME-FIB"
        elif n in self.fibonacci_set:
            return "FIBONACCI"
        elif n in self.prime_set:
            return "PRIME"
        else:
            return "COMPOSITE"
    
    def generate_complete_statistics(self) -> Dict:
        """生成完整的分类统计"""
        
        # 按类别统计
        classification_counts = defaultdict(int)
        classification_examples = defaultdict(list)
        classification_ranges = defaultdict(list)
        
        for n in range(1, self.max_theory + 1):
            cls = self.classify_theory(n)
            classification_counts[cls] += 1
            
            # 收集前10个示例
            if len(classification_examples[cls]) < 10:
                classification_examples[cls].append(n)
        
        total = self.max_theory
        
        # 计算理论密度
        densities = self._calculate_densities()
        
        # 分析特殊子类
        special_analysis = self._analyze_special_subcategories()
        
        # 分布分析
        distribution_analysis = self._analyze_distribution_patterns()
        
        return {
            'meta': {
                'total_theories': total,
                'max_theory': self.max_theory,
                'analysis_date': 'v3.0'
            },
            'classification_counts': dict(classification_counts),
            'classification_percentages': {
                cls: (count / total) * 100 
                for cls, count in classification_counts.items()
            },
            'classification_examples': dict(classification_examples),
            'theoretical_densities': densities,
            'special_analysis': special_analysis,
            'distribution_patterns': distribution_analysis,
            'comparative_analysis': self._comparative_analysis(classification_counts, total)
        }
    
    def _calculate_densities(self) -> Dict:
        """计算理论密度"""
        n = self.max_theory
        
        # 素数密度（素数定理）
        prime_density_theoretical = n / math.log(n) if n > 1 else 0
        prime_density_actual = len(self.primes)
        
        # Fibonacci密度
        fibonacci_density_actual = len([f for f in self.fibonacci_set if f <= n])
        fibonacci_density_theoretical = math.log(n) / math.log(self.phi) if n > 1 else 0
        
        # PRIME-FIB密度（交集）
        prime_fib_actual = len([x for x in range(1, n+1) 
                               if x in self.prime_set and x in self.fibonacci_set])
        
        return {
            'prime': {
                'theoretical': prime_density_theoretical,
                'actual': prime_density_actual,
                'accuracy': prime_density_actual / prime_density_theoretical if prime_density_theoretical > 0 else 0
            },
            'fibonacci': {
                'theoretical': fibonacci_density_theoretical,
                'actual': fibonacci_density_actual,
                'accuracy': fibonacci_density_actual / fibonacci_density_theoretical if fibonacci_density_theoretical > 0 else 0
            },
            'prime_fib_intersection': {
                'actual': prime_fib_actual,
                'rarity_factor': n / prime_fib_actual if prime_fib_actual > 0 else float('inf')
            }
        }
    
    def _analyze_special_subcategories(self) -> Dict:
        """分析特殊子类别"""
        
        # 特殊素数类型
        twin_primes = []
        mersenne_primes = []
        sophie_germain_primes = []
        
        for p in self.primes:
            if p <= self.max_theory:
                if self.prime_checker.is_twin_prime(p):
                    twin_primes.append(p)
                if self.prime_checker.is_mersenne_prime(p):
                    mersenne_primes.append(p)
                if self.prime_checker.is_sophie_germain_prime(p):
                    sophie_germain_primes.append(p)
        
        # 分析Fibonacci数的素因子结构
        fibonacci_factorizations = {}
        for fib in self.fibonacci_set:
            if fib <= self.max_theory and fib > 1:
                factors = self.prime_checker.prime_factorize(fib)
                fibonacci_factorizations[fib] = factors
        
        # 分析PRIME-FIB理论的稀有性
        prime_fib_theories = [n for n in range(1, self.max_theory + 1)
                             if n in self.prime_set and n in self.fibonacci_set]
        
        return {
            'special_prime_types': {
                'twin_primes': {
                    'count': len(twin_primes),
                    'examples': twin_primes[:10],
                    'percentage': len(twin_primes) / len(self.primes) * 100
                },
                'mersenne_primes': {
                    'count': len(mersenne_primes),
                    'examples': mersenne_primes,
                    'percentage': len(mersenne_primes) / len(self.primes) * 100
                },
                'sophie_germain_primes': {
                    'count': len(sophie_germain_primes),
                    'examples': sophie_germain_primes[:10],
                    'percentage': len(sophie_germain_primes) / len(self.primes) * 100
                }
            },
            'fibonacci_factorizations': fibonacci_factorizations,
            'prime_fib_rarity': {
                'total_count': len(prime_fib_theories),
                'theories': prime_fib_theories,
                'gaps_between': [prime_fib_theories[i+1] - prime_fib_theories[i] 
                               for i in range(len(prime_fib_theories)-1)] if len(prime_fib_theories) > 1 else [],
                'average_gap': sum([prime_fib_theories[i+1] - prime_fib_theories[i] 
                                  for i in range(len(prime_fib_theories)-1)]) / (len(prime_fib_theories)-1) if len(prime_fib_theories) > 1 else 0
            }
        }
    
    def _analyze_distribution_patterns(self) -> Dict:
        """分析分布模式"""
        
        # 分段分析
        segments = [
            (1, 50, "Early Universe"),
            (51, 100, "Foundation"),
            (101, 200, "Expansion"),
            (201, 500, "Complexity"),
            (501, 997, "Advanced")
        ]
        
        segment_analysis = {}
        for start, end, name in segments:
            segment_stats = defaultdict(int)
            for n in range(start, min(end + 1, self.max_theory + 1)):
                cls = self.classify_theory(n)
                segment_stats[cls] += 1
            
            total_in_segment = sum(segment_stats.values())
            segment_analysis[name] = {
                'range': f"T{start}-T{end}",
                'counts': dict(segment_stats),
                'percentages': {cls: (count / total_in_segment) * 100 
                              for cls, count in segment_stats.items()} if total_in_segment > 0 else {},
                'dominant_type': max(segment_stats, key=segment_stats.get) if segment_stats else None
            }
        
        return {
            'segment_analysis': segment_analysis,
            'clustering_analysis': self._analyze_clustering()
        }
    
    def _analyze_clustering(self) -> Dict:
        """分析聚类模式"""
        
        # 分析连续相同类型的理论
        clusters = defaultdict(list)
        current_cluster = []
        current_type = None
        
        for n in range(1, self.max_theory + 1):
            cls = self.classify_theory(n)
            
            if cls == current_type:
                current_cluster.append(n)
            else:
                if current_cluster and len(current_cluster) > 1:
                    clusters[current_type].append(current_cluster.copy())
                current_cluster = [n]
                current_type = cls
        
        # 添加最后一个聚类
        if current_cluster and len(current_cluster) > 1:
            clusters[current_type].append(current_cluster)
        
        # 统计聚类信息
        cluster_stats = {}
        for cls, cluster_list in clusters.items():
            if cluster_list:
                cluster_sizes = [len(cluster) for cluster in cluster_list]
                cluster_stats[cls] = {
                    'cluster_count': len(cluster_list),
                    'max_cluster_size': max(cluster_sizes),
                    'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes),
                    'total_clustered': sum(cluster_sizes),
                    'largest_clusters': sorted(cluster_list, key=len, reverse=True)[:3]
                }
        
        return cluster_stats
    
    def _comparative_analysis(self, counts: Dict, total: int) -> Dict:
        """比较分析"""
        
        # 与理论预期的比较
        expected_ratios = {
            'AXIOM': 1 / total,  # 只有1个
            'PRIME-FIB': 0.006,  # 极稀有
            'FIBONACCI': 0.015,  # 稀有
            'PRIME': 0.15,       # 根据素数定理
            'COMPOSITE': 0.829   # 大部分
        }
        
        actual_ratios = {cls: count / total for cls, count in counts.items()}
        
        ratio_comparison = {}
        for cls in expected_ratios:
            expected = expected_ratios[cls]
            actual = actual_ratios.get(cls, 0)
            ratio_comparison[cls] = {
                'expected': expected,
                'actual': actual,
                'deviation': abs(actual - expected) / expected if expected > 0 else 0,
                'ratio': actual / expected if expected > 0 else 0
            }
        
        return {
            'ratio_comparison': ratio_comparison,
            'diversity_index': self._calculate_diversity_index(counts, total),
            'mathematical_consistency': self._check_mathematical_consistency(counts)
        }
    
    def _calculate_diversity_index(self, counts: Dict, total: int) -> float:
        """计算多样性指数（Shannon熵）"""
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def _check_mathematical_consistency(self, counts: Dict) -> Dict:
        """检查数学一致性"""
        
        # 检查总数
        total_from_counts = sum(counts.values())
        expected_total = self.max_theory
        
        # 检查AXIOM数量
        axiom_correct = counts.get('AXIOM', 0) == 1
        
        # 检查素数总数
        prime_total = counts.get('PRIME', 0) + counts.get('PRIME-FIB', 0)
        expected_primes = len(self.primes)
        
        # 检查Fibonacci总数
        fib_total = counts.get('FIBONACCI', 0) + counts.get('PRIME-FIB', 0)
        expected_fibs = len([f for f in self.fibonacci_set if f <= self.max_theory])
        
        return {
            'total_consistency': total_from_counts == expected_total,
            'axiom_consistency': axiom_correct,
            'prime_consistency': prime_total == expected_primes,
            'fibonacci_consistency': fib_total == expected_fibs,
            'details': {
                'total': {'actual': total_from_counts, 'expected': expected_total},
                'primes': {'actual': prime_total, 'expected': expected_primes},
                'fibonacci': {'actual': fib_total, 'expected': expected_fibs}
            }
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """生成详细报告"""
        
        stats = self.generate_complete_statistics()
        
        report = []
        report.append("🔬 T{n}理论五类分类统计报告 v3.0")
        report.append("=" * 60)
        
        # 基本统计
        report.append(f"\n📊 基本统计 (T1-T{self.max_theory})")
        report.append("-" * 40)
        report.append(f"总理论数: {stats['meta']['total_theories']}")
        
        for cls, count in stats['classification_counts'].items():
            percentage = stats['classification_percentages'][cls]
            examples = ", ".join(f"T{x}" for x in stats['classification_examples'][cls][:5])
            report.append(f"{cls:12s}: {count:4d} ({percentage:5.2f}%) - 示例: {examples}")
        
        # 密度分析
        report.append(f"\n🎯 理论密度分析")
        report.append("-" * 40)
        densities = stats['theoretical_densities']
        
        report.append(f"素数密度:")
        report.append(f"  理论值: {densities['prime']['theoretical']:.1f}")
        report.append(f"  实际值: {densities['prime']['actual']}")
        report.append(f"  准确度: {densities['prime']['accuracy']*100:.1f}%")
        
        report.append(f"Fibonacci密度:")
        report.append(f"  理论值: {densities['fibonacci']['theoretical']:.1f}")
        report.append(f"  实际值: {densities['fibonacci']['actual']}")
        
        report.append(f"PRIME-FIB稀有度:")
        report.append(f"  总数: {densities['prime_fib_intersection']['actual']}")
        report.append(f"  稀有因子: {densities['prime_fib_intersection']['rarity_factor']:.1f}")
        
        # 特殊分析
        report.append(f"\n⭐ 特殊理论分析")
        report.append("-" * 40)
        special = stats['special_analysis']
        
        # PRIME-FIB理论
        prime_fib_info = special['prime_fib_rarity']
        report.append(f"PRIME-FIB双重理论: {prime_fib_info['theories']}")
        if prime_fib_info['gaps_between']:
            report.append(f"理论间平均间隔: {prime_fib_info['average_gap']:.1f}")
        
        # 特殊素数类型
        report.append(f"\n特殊素数类型:")
        for prime_type, info in special['special_prime_types'].items():
            if info['count'] > 0:
                report.append(f"  {prime_type}: {info['count']}个 ({info['percentage']:.1f}%)")
                report.append(f"    示例: {info['examples'][:5]}")
        
        # 分段分析
        report.append(f"\n📈 分段分布分析")
        report.append("-" * 40)
        for segment_name, segment_data in stats['distribution_patterns']['segment_analysis'].items():
            report.append(f"{segment_name} {segment_data['range']}:")
            report.append(f"  主导类型: {segment_data['dominant_type']}")
            for cls, percentage in segment_data['percentages'].items():
                if percentage > 5:  # 只显示>5%的类型
                    report.append(f"  {cls}: {percentage:.1f}%")
        
        # 数学一致性
        report.append(f"\n✅ 数学一致性验证")
        report.append("-" * 40)
        consistency = stats['comparative_analysis']['mathematical_consistency']
        report.append(f"总数一致性: {'✅' if consistency['total_consistency'] else '❌'}")
        report.append(f"公理一致性: {'✅' if consistency['axiom_consistency'] else '❌'}")
        report.append(f"素数一致性: {'✅' if consistency['prime_consistency'] else '❌'}")
        report.append(f"Fibonacci一致性: {'✅' if consistency['fibonacci_consistency'] else '❌'}")
        
        # 多样性指数
        diversity = stats['comparative_analysis']['diversity_index']
        report.append(f"\n多样性指数 (Shannon熵): {diversity:.3f}")
        report.append(f"理论最大熵: {math.log2(5):.3f} (5类分类)")
        report.append(f"多样性比率: {diversity / math.log2(5) * 100:.1f}%")
        
        report_text = "\n".join(report)
        
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(report_text, encoding='utf-8')
            print(f"报告已保存到: {output_file}")
        
        return report_text

def main():
    """主函数"""
    stats_generator = ClassificationStatistics(997)
    
    print("正在生成T{n}理论分类统计报告...")
    report = stats_generator.generate_report("classification_statistics_report.txt")
    
    print(report)
    
    # 生成JSON格式的详细数据
    import json
    detailed_stats = stats_generator.generate_complete_statistics()
    
    with open("classification_statistics_data.json", "w", encoding='utf-8') as f:
        json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细数据已保存到: classification_statistics_data.json")

if __name__ == "__main__":
    main()