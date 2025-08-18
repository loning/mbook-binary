#!/usr/bin/env python3
"""
理论依赖结构分析器
分析T1-T150理论的依赖关系，验证DAG结构的逻辑一致性
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

class TheoryDependencyAnalyzer:
    def __init__(self, base_path: str = "src/existencephilosophy/theorems"):
        self.base_path = Path(base_path)
        self.theories = {}  # theory_id -> {name, dependencies, derivation_basis}
        self.dependency_graph = defaultdict(set)  # theory -> set of dependencies
        self.reverse_graph = defaultdict(set)  # theory -> set of dependents
        
    def extract_theory_info(self, filepath: Path) -> Dict:
        """从理论文件中提取信息"""
        theory_num = None
        theory_name = None
        dependencies = set()
        derivation_basis = None
        
        # 从文件名提取理论编号
        filename = filepath.name
        match = re.match(r'T(\d+)', filename)
        if match:
            theory_num = int(match.group(1))
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 提取理论名称
                name_match = re.search(r'#\s*T\d+[：:]\s*(.+?)[\(\(]', content)
                if name_match:
                    theory_name = name_match.group(1).strip()
                
                # 提取推导依据 - 尝试多种格式
                basis_patterns = [
                    r'推导依据\s*\n(.+?)(?:##\s*依赖理论|##\s*形式化)',
                    r'推导基础\s*\n(.+?)(?:##\s*依赖理论|##\s*形式化)',
                    r'##\s*推导依据\s*\n(.+?)(?:##|$)',
                    r'##\s*推导基础\s*\n(.+?)(?:##|$)',
                ]
                
                for pattern in basis_patterns:
                    basis_match = re.search(pattern, content, re.DOTALL)
                    if basis_match:
                        derivation_basis = basis_match.group(1).strip()
                        # 从推导依据中提取理论依赖（包括括号中的）
                        theory_refs = re.findall(r'[TAC]\d+', basis_match.group(1))
                        dependencies.update(theory_refs)
                        break
                
                # 提取依赖理论部分
                dep_patterns = [
                    r'##\s*依赖理论\s*\n(.+?)(?:##|$)',
                    r'依赖理论[：:]\s*\n(.+?)(?:##|$)',
                ]
                
                for pattern in dep_patterns:
                    dep_section = re.search(pattern, content, re.DOTALL)
                    if dep_section:
                        # 查找所有理论引用
                        theory_refs = re.findall(r'[TAC]\d+', dep_section.group(1))
                        dependencies.update(theory_refs)
                        break
                
                # 额外搜索全文中的理论引用（作为备用）
                if not dependencies:
                    all_refs = re.findall(r'(?:根据|由|基于|来自|依据)[TAC]\d+', content)
                    for ref in all_refs:
                        theory_refs = re.findall(r'[TAC]\d+', ref)
                        dependencies.update(theory_refs)
                        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
        return {
            'id': f'T{theory_num}' if theory_num else None,
            'name': theory_name,
            'dependencies': dependencies,
            'derivation_basis': derivation_basis,
            'filepath': str(filepath)
        }
    
    def scan_all_theories(self):
        """扫描所有理论文件"""
        print("\n=== 扫描理论文件 ===")
        
        # 获取所有T*.md文件
        theory_files = sorted(self.base_path.glob("T*.md"))
        
        for filepath in theory_files:
            info = self.extract_theory_info(filepath)
            if info['id']:
                self.theories[info['id']] = info
                # 构建依赖图
                for dep in info['dependencies']:
                    self.dependency_graph[info['id']].add(dep)
                    self.reverse_graph[dep].add(info['id'])
                    
        print(f"扫描完成：找到 {len(self.theories)} 个理论")
        
    def check_circular_dependencies(self) -> List[List[str]]:
        """检查循环依赖（使用DFS）"""
        print("\n=== 检查循环依赖 ===")
        
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # 找到循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True
                    
            path.pop()
            rec_stack.remove(node)
            return False
        
        for theory in self.theories:
            if theory not in visited:
                dfs(theory)
                
        if cycles:
            print(f"⚠️ 发现 {len(cycles)} 个循环依赖！")
            for cycle in cycles:
                print(f"  循环: {' -> '.join(cycle)}")
        else:
            print("✓ 没有发现循环依赖")
            
        return cycles
    
    def analyze_foundation_coverage(self):
        """分析基础理论覆盖度"""
        print("\n=== 基础理论覆盖度分析 ===")
        
        # 定义基础理论和公理
        foundations = set([f'A{i}' for i in range(1, 6)])  # A1-A5
        foundations.update([f'T{i}' for i in range(1, 51)])  # T1-T50
        
        # 分析T51-T150的基础覆盖
        advanced_theories = [f'T{i}' for i in range(51, 151)]
        
        missing_foundation = []
        weak_foundation = []
        
        for theory_id in advanced_theories:
            if theory_id not in self.theories:
                continue
                
            deps = self.theories[theory_id]['dependencies']
            foundation_deps = deps.intersection(foundations)
            
            if not deps:
                missing_foundation.append(theory_id)
            elif not foundation_deps:
                weak_foundation.append((theory_id, deps))
                
        print(f"分析范围: T51-T150 ({len(advanced_theories)} 个理论)")
        print(f"✓ 有基础依赖: {len(advanced_theories) - len(missing_foundation) - len(weak_foundation)} 个")
        
        if missing_foundation:
            print(f"⚠️ 缺少依赖: {len(missing_foundation)} 个")
            for t in missing_foundation[:5]:  # 只显示前5个
                print(f"  - {t}: {self.theories.get(t, {}).get('name', 'Unknown')}")
                
        if weak_foundation:
            print(f"⚠️ 仅依赖高级理论: {len(weak_foundation)} 个")
            for t, deps in weak_foundation[:5]:
                print(f"  - {t}: 依赖 {deps}")
                
    def analyze_domain_coherence(self):
        """分析领域内部逻辑进展"""
        print("\n=== 领域内部逻辑进展分析 ===")
        
        domains = {
            "模态与可能性": list(range(51, 61)),
            "现象学与意识": list(range(61, 71)),
            "语言哲学": list(range(71, 81)),
            "政治哲学": list(range(81, 91)),
            "美学哲学": list(range(91, 101)),
            "科学哲学": list(range(101, 111)),
            "技术哲学": list(range(111, 121)),
            "环境哲学": list(range(121, 131)),
            "生命伦理学": list(range(131, 141)),
            "文明哲学": list(range(141, 151))
        }
        
        for domain_name, theory_nums in domains.items():
            print(f"\n{domain_name} (T{theory_nums[0]}-T{theory_nums[-1]}):")
            
            # 检查域内依赖
            internal_deps = 0
            external_deps = 0
            foundation_deps = 0
            
            for num in theory_nums:
                theory_id = f'T{num}'
                if theory_id not in self.theories:
                    continue
                    
                deps = self.theories[theory_id]['dependencies']
                for dep in deps:
                    if dep.startswith('T'):
                        dep_num = int(dep[1:])
                        if dep_num in theory_nums:
                            internal_deps += 1
                        elif dep_num <= 50:
                            foundation_deps += 1
                        else:
                            external_deps += 1
                    elif dep.startswith('A'):
                        foundation_deps += 1
                        
            total_deps = internal_deps + external_deps + foundation_deps
            if total_deps > 0:
                print(f"  内部依赖: {internal_deps} ({internal_deps*100//total_deps}%)")
                print(f"  基础依赖: {foundation_deps} ({foundation_deps*100//total_deps}%)")
                print(f"  跨域依赖: {external_deps} ({external_deps*100//total_deps}%)")
                
    def analyze_cross_domain_integration(self):
        """分析跨领域整合关系"""
        print("\n=== 跨领域整合关系分析 ===")
        
        # 定义领域
        domains = {
            "基础存在论": (1, 10),
            "认识价值论": (11, 25),
            "高级本体论": (26, 40),
            "前沿应用": (41, 50),
            "模态形而上学": (51, 60),
            "认知哲学": (61, 70),
            "语言哲学": (71, 80),
            "政治哲学": (81, 90),
            "美学哲学": (91, 100),
            "科学哲学": (101, 110),
            "技术哲学": (111, 120),
            "环境哲学": (121, 130),
            "生命伦理学": (131, 140),
            "文明哲学": (141, 150)
        }
        
        # 构建跨域依赖矩阵
        cross_domain_deps = defaultdict(lambda: defaultdict(int))
        
        for theory_id, info in self.theories.items():
            if not theory_id.startswith('T'):
                continue
                
            theory_num = int(theory_id[1:])
            theory_domain = None
            
            for domain, (start, end) in domains.items():
                if start <= theory_num <= end:
                    theory_domain = domain
                    break
                    
            if not theory_domain:
                continue
                
            for dep in info['dependencies']:
                if dep.startswith('T'):
                    dep_num = int(dep[1:])
                    dep_domain = None
                    
                    for domain, (start, end) in domains.items():
                        if start <= dep_num <= end:
                            dep_domain = domain
                            break
                            
                    if dep_domain and dep_domain != theory_domain:
                        cross_domain_deps[theory_domain][dep_domain] += 1
                        
        # 显示主要跨域依赖
        print("\n主要跨域依赖关系:")
        for source_domain, targets in cross_domain_deps.items():
            if targets:
                top_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)[:3]
                deps_str = ", ".join([f"{d}({c})" for d, c in top_targets])
                print(f"  {source_domain} → {deps_str}")
                
    def analyze_terminus_consistency(self):
        """分析T150终极理论的收敛性"""
        print("\n=== T150终极理论收敛性分析 ===")
        
        if 'T150' not in self.theories:
            print("⚠️ T150理论文件未找到")
            return
            
        # 分析到达T150的路径
        def find_paths_to_target(target, max_depth=10):
            """找到所有到达目标理论的依赖路径"""
            paths = []
            
            def dfs(current, path, depth):
                if depth > max_depth:
                    return
                    
                if current == target:
                    paths.append(list(path))
                    return
                    
                for dependent in self.reverse_graph.get(current, []):
                    if dependent not in path:  # 避免循环
                        path.append(dependent)
                        dfs(dependent, path, depth + 1)
                        path.pop()
                        
            # 从基础理论和公理开始
            starts = [f'A{i}' for i in range(1, 6)]
            starts.extend([f'T{i}' for i in range(1, 11)])
            
            for start in starts:
                if start in self.theories or start.startswith('A'):
                    dfs(start, [start], 0)
                    
            return paths
            
        t150_info = self.theories['T150']
        print(f"T150: {t150_info['name']}")
        print(f"直接依赖: {t150_info['dependencies']}")
        
        # 计算理论深度（从基础到T150的最短路径）
        def calculate_depth():
            depth = defaultdict(lambda: float('inf'))
            queue = deque()
            
            # 基础理论深度为0
            for i in range(1, 6):
                depth[f'A{i}'] = 0
                queue.append(f'A{i}')
            for i in range(1, 11):
                depth[f'T{i}'] = 1
                queue.append(f'T{i}')
                
            while queue:
                current = queue.popleft()
                for dependent in self.reverse_graph.get(current, []):
                    if depth[dependent] > depth[current] + 1:
                        depth[dependent] = depth[current] + 1
                        queue.append(dependent)
                        
            return depth
            
        depths = calculate_depth()
        
        if 'T150' in depths and depths['T150'] < float('inf'):
            print(f"理论深度: {depths['T150']} 层")
            
            # 分析关键依赖路径
            key_dependencies = set()
            for dep in t150_info['dependencies']:
                if dep in depths:
                    key_dependencies.add((dep, depths[dep]))
                    
            print("关键依赖层次:")
            for dep, d in sorted(key_dependencies, key=lambda x: x[1]):
                dep_name = self.theories.get(dep, {}).get('name', dep)
                print(f"  {dep} (深度{d}): {dep_name}")
        else:
            print("⚠️ T150无法从基础理论到达")
            
    def generate_report(self):
        """生成综合分析报告"""
        print("\n" + "="*60)
        print("理论体系结构分析报告")
        print("="*60)
        
        # 基本统计
        print(f"\n理论总数: {len(self.theories)}")
        
        # 依赖统计
        total_deps = sum(len(deps) for deps in self.dependency_graph.values())
        avg_deps = total_deps / len(self.theories) if self.theories else 0
        print(f"总依赖关系: {total_deps}")
        print(f"平均依赖数: {avg_deps:.2f}")
        
        # 检查没有推导基础的理论
        no_basis = [t for t, info in self.theories.items() 
                    if not info['dependencies'] and int(t[1:]) > 50]
        if no_basis:
            print(f"\n⚠️ 缺少推导基础的理论 (T51-T150): {len(no_basis)} 个")
            for t in no_basis[:10]:  # 显示前10个
                print(f"  - {t}: {self.theories[t]['name']}")
                
        # 最多被依赖的理论
        most_depended = sorted(self.reverse_graph.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:5]
        print("\n最多被依赖的理论:")
        for theory, dependents in most_depended:
            theory_name = self.theories.get(theory, {}).get('name', theory)
            print(f"  {theory} ({theory_name}): 被 {len(dependents)} 个理论依赖")
            
        print("\n" + "="*60)

def main():
    analyzer = TheoryDependencyAnalyzer()
    
    # 执行分析步骤
    analyzer.scan_all_theories()
    analyzer.check_circular_dependencies()
    analyzer.analyze_foundation_coverage()
    analyzer.analyze_domain_coherence()
    analyzer.analyze_cross_domain_integration()
    analyzer.analyze_terminus_consistency()
    analyzer.generate_report()

if __name__ == "__main__":
    main()