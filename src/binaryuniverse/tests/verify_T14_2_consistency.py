#!/usr/bin/env python3
"""
验证T14-2三个文件的一致性：
1. 理论文件：T14-2-phi-standard-model-unification.md
2. 形式化文件：formal/T14-2-formal.md  
3. 测试文件：tests/test_T14_2.py
"""

import re
from typing import List, Set, Dict, Tuple

class ConsistencyChecker:
    """T14-2文件一致性检查器"""
    
    def __init__(self):
        self.theory_file = "/Users/cookie/the-binarymath/docs/binaryuniverse/T14-2-phi-standard-model-unification.md"
        self.formal_file = "/Users/cookie/the-binarymath/docs/binaryuniverse/formal/T14-2-formal.md"
        self.test_file = "/Users/cookie/the-binarymath/docs/binaryuniverse/tests/test_T14_2.py"
        
    def extract_key_concepts(self, content: str) -> Dict[str, List[str]]:
        """提取关键概念"""
        concepts = {
            "observer_entanglement": [],
            "recursive_depth": [],
            "chirality": [],
            "anomaly_cancellation": [],
            "coupling_constants": [],
            "three_generations": []
        }
        
        # 观察者纠缠
        if "观察者-系统纠缠" in content or "ObserverSystemEntanglement" in content:
            concepts["observer_entanglement"].append("found")
        if "MeasuredValue(O, ψ_obs)" in content or "Entangle(SystemState(O), ψ_obs)" in content:
            concepts["observer_entanglement"].append("formula")
        if "α ≈ 1/137" in content and ("观察者" in content or "observer" in content.lower()):
            concepts["observer_entanglement"].append("alpha_example")
            
        # 递归深度
        for match in re.finditer(r"n\s*=\s*(\d)", content):
            depth = match.group(1)
            if depth == "0" and ("强" in content or "strong" in content.lower()):
                concepts["recursive_depth"].append("n=0_strong")
            elif depth == "1" and ("电磁" in content or "electromagnetic" in content.lower()):
                concepts["recursive_depth"].append("n=1_em")
            elif depth == "2" and ("弱" in content or "weak" in content.lower()):
                concepts["recursive_depth"].append("n=2_weak")
                
        # 手性
        if "LEFT" in content or "RIGHT" in content or "左手" in content or "右手" in content:
            concepts["chirality"].append("found")
        if "Y = 1/3" in content and ("左手夸克" in content or "left.*quark" in content.lower()):
            concepts["chirality"].append("left_quark_hypercharge")
        if "Y = 4/3" in content and ("右手.*上夸克" in content or "right.*up" in content.lower()):
            concepts["chirality"].append("right_up_hypercharge")
            
        # 反常消除
        if "anomaly" in content.lower() or "反常" in content:
            concepts["anomaly_cancellation"].append("found")
        if "left.*right" in content.lower() and ("cancel" in content.lower() or "消除" in content):
            concepts["anomaly_cancellation"].append("left_right_cancellation")
            
        # 耦合常数
        if "g = e / sin(θ_W)" in content or "g = e/sin(θ_W)" in content:
            concepts["coupling_constants"].append("weinberg_relation")
        if "sin²θ_W" in content and ("0.23" in content or "0.2312" in content):
            concepts["coupling_constants"].append("weinberg_value")
            
        # 三代结构
        if "三代" in content or "three generation" in content.lower():
            concepts["three_generations"].append("found")
        if "第四代" in content and ("违反" in content or "violate" in content.lower()):
            concepts["three_generations"].append("no_fourth")
            
        return concepts
    
    def check_consistency(self) -> Dict[str, bool]:
        """检查三个文件的一致性"""
        # 读取文件
        with open(self.theory_file, 'r', encoding='utf-8') as f:
            theory_content = f.read()
        with open(self.formal_file, 'r', encoding='utf-8') as f:
            formal_content = f.read()
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_content = f.read()
            
        # 提取概念
        theory_concepts = self.extract_key_concepts(theory_content)
        formal_concepts = self.extract_key_concepts(formal_content)
        test_concepts = self.extract_key_concepts(test_content)
        
        # 检查一致性
        consistency = {}
        
        # 1. 观察者效应
        observer_check = (
            len(theory_concepts["observer_entanglement"]) > 0 and
            len(formal_concepts["observer_entanglement"]) > 0 and
            ("observer" in test_content.lower() or "观察者" in test_content)
        )
        consistency["observer_effect"] = observer_check
        
        # 2. 递归深度层次
        depth_check = (
            "n=0_strong" in theory_concepts["recursive_depth"] and
            "n=0_strong" in formal_concepts["recursive_depth"] and
            "n=0_strong" in test_concepts["recursive_depth"]
        )
        consistency["recursive_depth_hierarchy"] = depth_check
        
        # 3. 手性结构
        chirality_check = (
            "found" in theory_concepts["chirality"] and
            "found" in formal_concepts["chirality"] and
            "found" in test_concepts["chirality"]
        )
        consistency["chirality_structure"] = chirality_check
        
        # 4. 反常消除
        anomaly_check = (
            "found" in theory_concepts["anomaly_cancellation"] and
            "found" in formal_concepts["anomaly_cancellation"] and
            "found" in test_concepts["anomaly_cancellation"]
        )
        consistency["anomaly_cancellation"] = anomaly_check
        
        # 5. Weinberg角
        weinberg_check = (
            (len(theory_concepts["coupling_constants"]) > 0 or "Weinberg" in theory_content) and
            "weinberg_relation" in formal_concepts["coupling_constants"] and
            "weinberg_value" in test_concepts["coupling_constants"]
        )
        consistency["weinberg_angle"] = weinberg_check
        
        # 6. 三代结构
        generation_check = (
            "found" in theory_concepts["three_generations"] and
            "found" in formal_concepts["three_generations"] and
            "found" in test_concepts["three_generations"]
        )
        consistency["three_generations"] = generation_check
        
        return consistency
    
    def print_report(self):
        """打印一致性报告"""
        consistency = self.check_consistency()
        
        print("T14-2 文件一致性检查报告")
        print("=" * 50)
        
        all_consistent = True
        for aspect, is_consistent in consistency.items():
            status = "✓" if is_consistent else "✗"
            print(f"{status} {aspect}: {'一致' if is_consistent else '不一致'}")
            if not is_consistent:
                all_consistent = False
                
        print("=" * 50)
        if all_consistent:
            print("结论：三个文件完全一致！")
        else:
            print("警告：发现不一致之处，需要进一步检查")
            
        # 详细概念分析
        print("\n详细分析：")
        with open(self.theory_file, 'r', encoding='utf-8') as f:
            theory_content = f.read()
        with open(self.formal_file, 'r', encoding='utf-8') as f:
            formal_content = f.read()
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_content = f.read()
            
        theory_concepts = self.extract_key_concepts(theory_content)
        formal_concepts = self.extract_key_concepts(formal_content)
        test_concepts = self.extract_key_concepts(test_content)
        
        print("\n关键概念覆盖情况：")
        for concept_type in theory_concepts:
            print(f"\n{concept_type}:")
            print(f"  理论文件: {theory_concepts[concept_type]}")
            print(f"  形式化文件: {formal_concepts[concept_type]}")
            print(f"  测试文件: {test_concepts[concept_type]}")

def main():
    """主程序"""
    checker = ConsistencyChecker()
    checker.print_report()

if __name__ == "__main__":
    main()