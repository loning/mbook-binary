#!/usr/bin/env python3
"""
二进制宇宙理论 - 集成测试套件

运行所有理论组件的集成测试，验证：
1. 所有定理测试通过
2. 理论依赖关系正确
3. 从公理到应用的推导链完整
4. 形式化描述与测试一致
"""

import unittest
import subprocess
import sys
import os
from typing import List, Dict, Tuple
import importlib.util

# 添加测试目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class IntegrationTestSuite(unittest.TestCase):
    """集成测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.test_modules = cls._discover_test_modules()
        cls.test_results = {}
        cls.dependency_graph = cls._build_dependency_graph()
        
    @classmethod
    def _discover_test_modules(cls) -> List[str]:
        """发现所有测试模块"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_files = []
        
        for file in sorted(os.listdir(test_dir)):
            if file.startswith('test_') and file.endswith('.py') and file != 'test_integration.py':
                module_name = file[:-3]
                test_files.append(module_name)
                
        return test_files
    
    @classmethod
    def _build_dependency_graph(cls) -> Dict[str, List[str]]:
        """构建依赖关系图"""
        dependencies = {
            # 基础层
            'test_philosophy': [],
            'test_A1': ['test_philosophy'],
            
            # 定义层
            'test_D1_1': ['test_A1'],
            'test_D1_2': ['test_D1_1'],
            'test_D1_3': ['test_D1_2'],
            'test_D1_4': ['test_D1_1'],
            'test_D1_5': ['test_D1_1', 'test_D1_4'],
            'test_D1_6': ['test_D1_1', 'test_D1_4'],
            'test_D1_7': ['test_D1_5'],
            'test_D1_8': ['test_D1_2', 'test_D1_3'],
            
            # 引理层
            'test_L1_1': ['test_D1_1'],
            'test_L1_2': ['test_L1_1'],
            'test_L1_3': ['test_L1_2'],
            'test_L1_4': ['test_L1_3'],
            'test_L1_5': ['test_D1_3', 'test_L1_4'],
            'test_L1_6': ['test_L1_5'],
            'test_L1_7': ['test_D1_5'],
            'test_L1_8': ['test_L1_7'],
            
            # 定理层 - T1-T2系列
            'test_T1_1': ['test_D1_1', 'test_D1_4', 'test_D1_6'],
            'test_T1_2': ['test_T1_1', 'test_L1_8'],
            'test_T2_1': ['test_L1_1'],
            'test_T2_2': ['test_T2_1', 'test_L1_6'],
            'test_T2_3': ['test_T2_2', 'test_D1_6'],
            'test_T2_4': ['test_L1_2'],
            'test_T2_5': ['test_L1_3'],
            'test_T2_6': ['test_T2_5', 'test_L1_4'],
            'test_T2_7': ['test_T2_6', 'test_L1_6'],
            'test_T2_10': ['test_T2_7', 'test_D1_8'],
            'test_T2_11': ['test_T2_10', 'test_T1_1'],
            
            # 定理层 - T3系列（量子）
            'test_T3_1': ['test_D1_2', 'test_D1_5', 'test_D1_8', 'test_L1_6', 'test_T2_7'],
            'test_T3_2': ['test_T3_1', 'test_D1_7'],
            'test_T3_3': ['test_T3_1', 'test_T3_2'],
            'test_T3_4': ['test_T3_3'],
            'test_T3_5': ['test_T3_1', 'test_T2_10'],
            
            # 定理层 - T4系列（数学结构）
            'test_T4_1': ['test_D1_1', 'test_D1_8', 'test_T2_10'],
            'test_T4_2': ['test_T4_1', 'test_T2_10'],
            'test_T4_3': ['test_T4_1', 'test_T4_2'],
            'test_T4_4': ['test_T4_1', 'test_T4_3'],
            
            # 定理层 - T5系列（信息理论）
            'test_T5_1': ['test_D1_6', 'test_L1_8', 'test_T1_1'],
            'test_T5_2': ['test_T5_1', 'test_T2_10'],
            'test_T5_3': ['test_T5_1', 'test_T5_2'],
            'test_T5_4': ['test_T5_3', 'test_T2_3'],
            'test_T5_5': ['test_T5_1', 'test_T3_5'],
            'test_T5_6': ['test_T5_1', 'test_T2_10'],
            'test_T5_7': ['test_T5_1', 'test_T1_1'],
            
            # 定理层 - T6系列（完备性）
            'test_T6_1': ['test_P5_1', 'test_T5_7'],
            'test_T6_2': ['test_P5_1', 'test_T5_7'],
            'test_T6_3': ['test_T6_2', 'test_T4_4'],
            
            # 推论层
            'test_C1_1': ['test_T2_2'],
            'test_C1_2': ['test_T2_3'],
            'test_C1_3': ['test_T2_3', 'test_T5_1'],
            'test_C2_1': ['test_T3_2'],
            'test_C2_2': ['test_T3_2', 'test_T5_3'],
            'test_C2_3': ['test_T5_1', 'test_T3_2'],
            'test_C3_1': ['test_T1_1', 'test_T2_11'],
            'test_C3_2': ['test_T3_1', 'test_T4_1'],
            'test_C3_3': ['test_T5_6', 'test_T4_3'],
            'test_C5_1': ['test_T3_5', 'test_T5_5'],
            'test_C5_2': ['test_T5_2', 'test_T2_11'],
            'test_C5_3': ['test_T3_1', 'test_T5_5'],
            
            # 命题层
            'test_P1_1': ['test_D1_1', 'test_L1_2'],
            'test_P2_1': ['test_T2_4', 'test_T2_3'],
            'test_P3_1': ['test_D1_1', 'test_T2_2'],
            'test_P4_1': ['test_T2_10', 'test_T2_6'],
            'test_P5_1': ['test_T5_1', 'test_T5_6', 'test_T5_7'],
        }
        
        return dependencies
    
    def test_01_individual_tests(self):
        """运行所有单独的测试模块"""
        print("\n" + "="*60)
        print("运行单独测试模块")
        print("="*60)
        
        failed_modules = []
        passed_modules = []
        
        for module_name in self.test_modules:
            print(f"\n测试 {module_name}...")
            
            try:
                # 运行测试模块
                result = subprocess.run(
                    [sys.executable, '-m', 'unittest', f'{module_name}', '-v'],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                if result.returncode == 0:
                    print(f"✓ {module_name} 通过")
                    passed_modules.append(module_name)
                    self.test_results[module_name] = True
                else:
                    print(f"✗ {module_name} 失败")
                    print(f"错误输出:\n{result.stderr}")
                    failed_modules.append(module_name)
                    self.test_results[module_name] = False
                    
            except Exception as e:
                print(f"✗ {module_name} 执行错误: {e}")
                failed_modules.append(module_name)
                self.test_results[module_name] = False
        
        # 汇总结果
        print("\n" + "-"*60)
        print(f"测试完成: {len(passed_modules)} 通过, {len(failed_modules)} 失败")
        
        if failed_modules:
            print(f"\n失败的模块: {', '.join(failed_modules)}")
            
        # 断言所有测试通过
        self.assertEqual(len(failed_modules), 0, 
                        f"有 {len(failed_modules)} 个测试模块失败")
    
    def test_02_dependency_order(self):
        """验证依赖关系的正确性"""
        print("\n" + "="*60)
        print("验证依赖关系")
        print("="*60)
        
        # 拓扑排序验证依赖关系
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(module):
            if module in temp_visited:
                raise ValueError(f"检测到循环依赖: {module}")
            if module in visited:
                return
                
            temp_visited.add(module)
            
            # 访问所有依赖
            for dep in self.dependency_graph.get(module, []):
                visit(dep)
                
            temp_visited.remove(module)
            visited.add(module)
            order.append(module)
        
        # 访问所有模块
        for module in self.test_modules:
            if module not in visited:
                visit(module)
        
        print(f"依赖关系验证通过，共 {len(order)} 个模块")
        print(f"拓扑排序: {' -> '.join(order[:5])} ... -> {' -> '.join(order[-5:])}")
        
        # 验证每个模块的依赖都在其之前
        for i, module in enumerate(order):
            deps = self.dependency_graph.get(module, [])
            for dep in deps:
                dep_index = order.index(dep)
                self.assertLess(dep_index, i, 
                              f"{module} 依赖 {dep}，但 {dep} 在其之后")
    
    def test_03_theory_completeness(self):
        """验证理论完备性"""
        print("\n" + "="*60)
        print("验证理论完备性")
        print("="*60)
        
        # 统计各类理论元素
        categories = {
            'philosophy': [],
            'axiom': [],
            'definition': [],
            'lemma': [],
            'theorem': [],
            'corollary': [],
            'proposition': []
        }
        
        for module in self.test_modules:
            if 'philosophy' in module:
                categories['philosophy'].append(module)
            elif module.startswith('test_A'):
                categories['axiom'].append(module)
            elif module.startswith('test_D'):
                categories['definition'].append(module)
            elif module.startswith('test_L'):
                categories['lemma'].append(module)
            elif module.startswith('test_T'):
                categories['theorem'].append(module)
            elif module.startswith('test_C'):
                categories['corollary'].append(module)
            elif module.startswith('test_P'):
                categories['proposition'].append(module)
        
        # 输出统计
        print("\n理论元素统计:")
        total = 0
        for category, modules in categories.items():
            count = len(modules)
            total += count
            print(f"  {category.capitalize()}: {count}")
        print(f"\n总计: {total} 个理论元素")
        
        # 验证完备性要求
        self.assertGreaterEqual(len(categories['axiom']), 1, "至少需要一个公理")
        self.assertGreaterEqual(len(categories['definition']), 8, "至少需要8个基础定义")
        self.assertGreaterEqual(len(categories['theorem']), 20, "至少需要20个核心定理")
        
        # 验证推导链完整性
        print("\n验证推导链完整性...")
        
        # 从公理开始的可达性分析
        reachable = set(['test_philosophy', 'test_A1'])
        changed = True
        
        while changed:
            changed = False
            for module, deps in self.dependency_graph.items():
                if module not in reachable and all(dep in reachable for dep in deps):
                    reachable.add(module)
                    changed = True
        
        # 验证所有模块都可从公理推导
        unreachable = set(self.test_modules) - reachable
        self.assertEqual(len(unreachable), 0, 
                        f"以下模块无法从公理推导: {unreachable}")
        
        print(f"✓ 所有 {len(self.test_modules)} 个理论元素都可从唯一公理推导")
    
    def test_04_formal_verification_consistency(self):
        """验证形式化描述与测试的一致性"""
        print("\n" + "="*60)
        print("验证形式化描述一致性")
        print("="*60)
        
        formal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'formal')
        
        # 检查每个测试模块都有对应的形式化描述
        missing_formal = []
        
        for module in self.test_modules:
            if module == 'test_philosophy':
                formal_file = 'philosophy-formal.md'
            elif module.startswith('test_'):
                # 提取理论元素ID
                element_id = module[5:]  # 去掉 'test_'
                element_id = element_id.replace('_', '-')
                formal_file = f'{element_id}-formal.md'
            else:
                continue
                
            formal_path = os.path.join(formal_dir, formal_file)
            if not os.path.exists(formal_path):
                missing_formal.append((module, formal_file))
        
        if missing_formal:
            print(f"\n缺少形式化描述的模块:")
            for module, formal in missing_formal:
                print(f"  {module} -> {formal}")
        else:
            print(f"✓ 所有 {len(self.test_modules)} 个测试模块都有对应的形式化描述")
        
        self.assertEqual(len(missing_formal), 0, 
                        f"有 {len(missing_formal)} 个模块缺少形式化描述")
    
    def test_05_performance_metrics(self):
        """性能指标测试"""
        print("\n" + "="*60)
        print("性能指标")
        print("="*60)
        
        # 统计测试覆盖率
        total_tests = len(self.test_modules)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        coverage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n测试覆盖率: {coverage:.1f}% ({passed_tests}/{total_tests})")
        
        # 理论深度分析
        max_depth = 0
        for module in self.test_modules:
            depth = self._calculate_dependency_depth(module)
            max_depth = max(max_depth, depth)
        
        print(f"最大理论深度: {max_depth} 层")
        
        # 依赖复杂度
        avg_deps = sum(len(deps) for deps in self.dependency_graph.values()) / len(self.dependency_graph)
        print(f"平均依赖数: {avg_deps:.1f}")
        
        # 验证性能指标
        self.assertGreaterEqual(coverage, 100.0, "测试覆盖率必须达到100%")
        self.assertLessEqual(max_depth, 15, "理论深度不应超过15层")
        self.assertLessEqual(avg_deps, 3.0, "平均依赖数不应超过3")
    
    def _calculate_dependency_depth(self, module: str, visited: set = None) -> int:
        """计算模块的依赖深度"""
        if visited is None:
            visited = set()
            
        if module in visited:
            return 0
            
        visited.add(module)
        
        deps = self.dependency_graph.get(module, [])
        if not deps:
            return 1
            
        max_dep_depth = 0
        for dep in deps:
            dep_depth = self._calculate_dependency_depth(dep, visited)
            max_dep_depth = max(max_dep_depth, dep_depth)
            
        return max_dep_depth + 1


def run_integration_tests():
    """运行集成测试"""
    print("\n" + "="*80)
    print("二进制宇宙理论 - 集成测试套件")
    print("="*80)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)