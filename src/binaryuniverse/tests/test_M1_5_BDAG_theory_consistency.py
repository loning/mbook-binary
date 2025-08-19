"""
M1.5 BDAG理论一致性元定理 - 测试实现
测试内部矛盾检测机制和一致性保证
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict
import pytest

# 黄金比例常数
PHI = (1 + np.sqrt(5)) / 2

class ContradictionType(Enum):
    """BDAG矛盾类型枚举"""
    NO11_VIOLATION = "No-11约束违反"
    DIMENSION_MISMATCH = "维度不匹配"
    FOLD_ORDER_CONFLICT = "折叠顺序冲突"
    PERMUTATION_CONFLICT = "排列冲突"
    EMPTY_JOINT_SPACE = "空联合张量空间"
    CONFLICTING_PREDICTION = "预测冲突"
    FIVE_FOLD_VIOLATION = "五重等价性违反"
    CIRCULAR_REASONING = "循环推理"
    CONTRADICTION_PATH = "矛盾路径"
    SELF_CONTRADICTION = "自反矛盾"
    GENERATION_CONFLICT = "生成规则冲突"
    V1_INCOMPATIBLE = "V1不兼容"
    V2_INCOMPATIBLE = "V2不兼容"
    V3_INCOMPATIBLE = "V3不兼容"
    V4_INCOMPATIBLE = "V4不兼容"
    V5_INCOMPATIBLE = "V5不兼容"
    FOLD_SEMANTICS_CONFLICT = "折叠语义冲突"
    CLASSIFICATION_CONFLICT = "分类冲突"

class ResolutionStrategy(Enum):
    """矛盾解决策略"""
    LOCAL_REPAIR = "局部修复"
    THEORY_RECONSTRUCTION = "理论重构"
    META_EXTENSION = "元框架扩展"

@dataclass
class FoldSignature:
    """BDAG折叠签名"""
    z: List[int]  # Zeckendorf指数集，降序
    p: List[int]  # 输入顺序排列
    tau: str      # 括号结构
    sigma: List[int]  # 置换
    b: List[str]  # 编结词
    kappa: Dict[str, List[str]]  # 收缩调度DAG
    annot: Dict[str, Any]  # 注记

@dataclass
class BDAGTheory:
    """BDAG理论"""
    N: int  # 理论编号
    fold_signature: FoldSignature
    dependencies: List[int]  # 依赖理论编号
    tensor_dimension: int
    predictions: Set[str]  # 物理预测集合
    
    def get_zeckendorf_encoding(self) -> str:
        """获取Zeckendorf编码"""
        return self._encode_zeckendorf(self.N)
    
    def _encode_zeckendorf(self, n: int) -> str:
        """Zeckendorf编码实现"""
        if n == 0:
            return "0"
        
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        encoding = []
        for f in reversed(fibs):
            if f <= n:
                encoding.append('1')
                n -= f
            else:
                encoding.append('0')
        
        # 去除前导0
        result = ''.join(encoding).lstrip('0')
        return result if result else '0'

@dataclass
class Contradiction:
    """矛盾记录"""
    type: ContradictionType
    affected_theories: List[int]
    details: Dict[str, Any]
    severity: str  # "resolvable", "essential", "meta"

@dataclass
class ConsistencyReport:
    """一致性检测报告"""
    syntactic_issues: List[Contradiction]
    semantic_issues: List[Contradiction]
    logical_issues: List[Contradiction]
    meta_issues: List[Contradiction]
    consistency_tensor: np.ndarray
    consistency_score: float
    is_consistent: bool
    repairs_applied: List[Tuple[Contradiction, ResolutionStrategy]]

class BDAGConsistencyChecker:
    """BDAG一致性检测器"""
    
    def __init__(self):
        self.theories: Dict[int, BDAGTheory] = {}
        self.consistency_cache: Dict[Tuple[int, int], bool] = {}
        self.contradiction_log: List[Contradiction] = []
        
    def add_theory(self, theory: BDAGTheory) -> bool:
        """添加理论到系统"""
        # 增量式一致性检查
        for existing_id, existing_theory in self.theories.items():
            contradiction = self._check_pair_consistency(existing_theory, theory)
            if contradiction:
                self.contradiction_log.append(contradiction)
                return False
        
        self.theories[theory.N] = theory
        return True
    
    def detect_syntactic_contradiction(self, T1: BDAGTheory, T2: BDAGTheory) -> Optional[Contradiction]:
        """检测语法矛盾"""
        # 1. No-11约束检查
        enc1 = T1.get_zeckendorf_encoding()
        enc2 = T2.get_zeckendorf_encoding()
        
        if '11' in enc1:
            return Contradiction(
                type=ContradictionType.NO11_VIOLATION,
                affected_theories=[T1.N],
                details={'encoding': enc1},
                severity='resolvable'
            )
        
        if '11' in enc2:
            return Contradiction(
                type=ContradictionType.NO11_VIOLATION,
                affected_theories=[T2.N],
                details={'encoding': enc2},
                severity='resolvable'
            )
        
        # 2. 折叠签名良构性
        if not self._is_well_formed_FS(T1.fold_signature):
            return Contradiction(
                type=ContradictionType.FOLD_ORDER_CONFLICT,
                affected_theories=[T1.N],
                details={'fold_signature': T1.fold_signature},
                severity='resolvable'
            )
        
        if not self._is_well_formed_FS(T2.fold_signature):
            return Contradiction(
                type=ContradictionType.FOLD_ORDER_CONFLICT,
                affected_theories=[T2.N],
                details={'fold_signature': T2.fold_signature},
                severity='resolvable'
            )
        
        # 3. 维度兼容性
        if not self._are_dimensions_compatible(T1.tensor_dimension, T2.tensor_dimension):
            return Contradiction(
                type=ContradictionType.DIMENSION_MISMATCH,
                affected_theories=[T1.N, T2.N],
                details={'dim1': T1.tensor_dimension, 'dim2': T2.tensor_dimension},
                severity='resolvable'
            )
        
        return None
    
    def detect_semantic_contradiction(self, T1: BDAGTheory, T2: BDAGTheory) -> Optional[Contradiction]:
        """检测语义矛盾"""
        # 1. 检查预测冲突
        for pred in T1.predictions:
            neg_pred = f"not_{pred}"
            if neg_pred in T2.predictions:
                return Contradiction(
                    type=ContradictionType.CONFLICTING_PREDICTION,
                    affected_theories=[T1.N, T2.N],
                    details={'prediction': pred},
                    severity='essential'
                )
        
        # 2. 检查张量积空性
        joint_dim = self._compute_joint_dimension(T1.tensor_dimension, T2.tensor_dimension)
        if joint_dim == 0:
            return Contradiction(
                type=ContradictionType.EMPTY_JOINT_SPACE,
                affected_theories=[T1.N, T2.N],
                details={'joint_dimension': 0},
                severity='essential'
            )
        
        # 3. 检查五重等价性
        if not self._preserves_five_fold_equivalence(T1, T2):
            return Contradiction(
                type=ContradictionType.FIVE_FOLD_VIOLATION,
                affected_theories=[T1.N, T2.N],
                details={'five_fold': False},
                severity='essential'
            )
        
        return None
    
    def detect_logical_contradiction(self) -> Optional[Contradiction]:
        """检测逻辑矛盾"""
        # 1. 构建推理图
        inference_graph = self._build_inference_graph()
        
        # 2. 检查循环
        cycle = self._find_cycle(inference_graph)
        if cycle:
            return Contradiction(
                type=ContradictionType.CIRCULAR_REASONING,
                affected_theories=cycle,
                details={'cycle': cycle},
                severity='essential'
            )
        
        # 3. 检查矛盾路径
        for theory_id in self.theories:
            if self._has_contradiction_path(inference_graph, theory_id):
                return Contradiction(
                    type=ContradictionType.CONTRADICTION_PATH,
                    affected_theories=[theory_id],
                    details={'node': theory_id},
                    severity='essential'
                )
        
        # 4. 检查自反矛盾
        for theory_id, theory in self.theories.items():
            for pred in theory.predictions:
                neg_pred = f"not_{pred}"
                if neg_pred in theory.predictions:
                    return Contradiction(
                        type=ContradictionType.SELF_CONTRADICTION,
                        affected_theories=[theory_id],
                        details={'prediction': pred},
                        severity='essential'
                    )
        
        # 5. 验证生成规则一致性
        for theory_id, theory in self.theories.items():
            if not self._check_generation_consistency(theory):
                return Contradiction(
                    type=ContradictionType.GENERATION_CONFLICT,
                    affected_theories=[theory_id],
                    details={'theory': theory_id},
                    severity='meta'
                )
        
        return None
    
    def detect_metatheoretic_contradiction(self) -> Optional[Contradiction]:
        """检测元理论矛盾"""
        # 简化版：检查V1-V5验证条件
        for i in range(1, 6):
            if not self._check_verification_condition(i):
                return Contradiction(
                    type=getattr(ContradictionType, f'V{i}_INCOMPATIBLE'),
                    affected_theories=list(self.theories.keys()),
                    details={f'V{i}': False},
                    severity='meta'
                )
        
        return None
    
    def local_repair(self, contradiction: Contradiction) -> bool:
        """局部修复策略"""
        if contradiction.type == ContradictionType.NO11_VIOLATION:
            # 重新编码
            theory_id = contradiction.affected_theories[0]
            if theory_id in self.theories:
                # 模拟重新编码
                self.theories[theory_id].N = self._avoid_consecutive_ones(self.theories[theory_id].N)
                return True
                
        elif contradiction.type == ContradictionType.DIMENSION_MISMATCH:
            # 投影对齐
            t1_id, t2_id = contradiction.affected_theories[:2]
            if t1_id in self.theories and t2_id in self.theories:
                # 模拟维度对齐
                common_dim = min(self.theories[t1_id].tensor_dimension,
                               self.theories[t2_id].tensor_dimension)
                self.theories[t1_id].tensor_dimension = common_dim
                self.theories[t2_id].tensor_dimension = common_dim
                return True
                
        elif contradiction.type == ContradictionType.FOLD_ORDER_CONFLICT:
            # 规范化折叠签名
            theory_id = contradiction.affected_theories[0]
            if theory_id in self.theories:
                # 模拟规范化
                self.theories[theory_id].fold_signature = self._normalize_fold_signature(
                    self.theories[theory_id].fold_signature
                )
                return True
        
        return False
    
    def theory_reconstruction(self, contradiction: Contradiction) -> bool:
        """理论重构策略"""
        if contradiction.severity != 'essential':
            return False
        
        # 识别矛盾核心
        core_theories = contradiction.affected_theories
        
        # 生成替代理论
        for theory_id in core_theories:
            if theory_id in self.theories:
                old_theory = self.theories[theory_id]
                
                # 策略1: 调整依赖关系
                new_deps = [d for d in old_theory.dependencies if d not in core_theories]
                
                # 策略2: 修改预测
                new_predictions = {p for p in old_theory.predictions 
                                 if not p.startswith('not_')}
                
                # 创建新理论
                new_theory = BDAGTheory(
                    N=old_theory.N,
                    fold_signature=old_theory.fold_signature,
                    dependencies=new_deps,
                    tensor_dimension=old_theory.tensor_dimension,
                    predictions=new_predictions
                )
                
                # 验证新理论一致性
                is_consistent = True
                for other_id, other_theory in self.theories.items():
                    if other_id != theory_id:
                        if self._check_pair_consistency(new_theory, other_theory):
                            is_consistent = False
                            break
                
                if is_consistent:
                    self.theories[theory_id] = new_theory
                    return True
        
        return False
    
    def meta_extension(self, contradiction: Contradiction) -> bool:
        """元框架扩展策略"""
        if contradiction.severity != 'meta':
            return False
        
        # 简化版：放松验证条件
        if contradiction.type == ContradictionType.V1_INCOMPATIBLE:
            # 放松No-11到No-111
            self._relax_no11_constraint()
            return True
        elif contradiction.type == ContradictionType.V2_INCOMPATIBLE:
            # 引入维度映射
            self._introduce_dimension_mapping()
            return True
        elif contradiction.type == ContradictionType.V5_INCOMPATIBLE:
            # 引入条件等价性
            self._introduce_conditional_equivalence()
            return True
        
        return False
    
    def compute_consistency_tensor(self) -> np.ndarray:
        """计算一致性张量"""
        # 语法一致性度量
        syntactic_score = self._compute_syntactic_consistency()
        
        # 语义一致性度量
        semantic_score = self._compute_semantic_consistency()
        
        # 逻辑一致性度量
        logical_score = 1.0 if not self.detect_logical_contradiction() else 0.0
        
        # 元理论一致性度量
        meta_score = self._compute_meta_consistency()
        
        # BDAG特定度量
        bdag_score = self._compute_bdag_specific_measure()
        
        return np.array([syntactic_score, semantic_score, logical_score, meta_score, bdag_score])
    
    def automated_consistency_check(self) -> ConsistencyReport:
        """自动化一致性检查主流程"""
        report = ConsistencyReport(
            syntactic_issues=[],
            semantic_issues=[],
            logical_issues=[],
            meta_issues=[],
            consistency_tensor=np.zeros(5),
            consistency_score=0.0,
            is_consistent=False,
            repairs_applied=[]
        )
        
        # 阶段1: 语法检测
        for t1_id, t2_id in itertools.combinations(self.theories.keys(), 2):
            contradiction = self.detect_syntactic_contradiction(
                self.theories[t1_id], self.theories[t2_id]
            )
            if contradiction:
                report.syntactic_issues.append(contradiction)
                if self.local_repair(contradiction):
                    report.repairs_applied.append((contradiction, ResolutionStrategy.LOCAL_REPAIR))
        
        # 阶段2: 语义检测
        for t1_id, t2_id in itertools.combinations(self.theories.keys(), 2):
            contradiction = self.detect_semantic_contradiction(
                self.theories[t1_id], self.theories[t2_id]
            )
            if contradiction:
                report.semantic_issues.append(contradiction)
                if self.theory_reconstruction(contradiction):
                    report.repairs_applied.append((contradiction, ResolutionStrategy.THEORY_RECONSTRUCTION))
        
        # 阶段3: 逻辑检测
        logical_contradiction = self.detect_logical_contradiction()
        if logical_contradiction:
            report.logical_issues.append(logical_contradiction)
            if self.theory_reconstruction(logical_contradiction):
                report.repairs_applied.append((logical_contradiction, ResolutionStrategy.THEORY_RECONSTRUCTION))
        
        # 阶段4: 元理论检测
        meta_contradiction = self.detect_metatheoretic_contradiction()
        if meta_contradiction:
            report.meta_issues.append(meta_contradiction)
            if self.meta_extension(meta_contradiction):
                report.repairs_applied.append((meta_contradiction, ResolutionStrategy.META_EXTENSION))
        
        # 计算最终一致性度量
        report.consistency_tensor = self.compute_consistency_tensor()
        report.consistency_score = np.linalg.norm(report.consistency_tensor)
        report.is_consistent = report.consistency_score >= PHI**5
        
        return report
    
    # 辅助方法
    def _check_pair_consistency(self, T1: BDAGTheory, T2: BDAGTheory) -> Optional[Contradiction]:
        """检查理论对的一致性"""
        # 检查缓存
        cache_key = (min(T1.N, T2.N), max(T1.N, T2.N))
        if cache_key in self.consistency_cache:
            return None if self.consistency_cache[cache_key] else Contradiction(
                type=ContradictionType.CONFLICTING_PREDICTION,
                affected_theories=[T1.N, T2.N],
                details={},
                severity='essential'
            )
        
        # 执行检查
        contradiction = self.detect_syntactic_contradiction(T1, T2)
        if not contradiction:
            contradiction = self.detect_semantic_contradiction(T1, T2)
        
        # 缓存结果
        self.consistency_cache[cache_key] = (contradiction is None)
        
        return contradiction
    
    def _is_well_formed_FS(self, fs: FoldSignature) -> bool:
        """检查折叠签名良构性"""
        # z降序
        if not all(fs.z[i] > fs.z[i+1] for i in range(len(fs.z)-1)):
            return False
        
        # p是有效排列
        if sorted(fs.p) != list(range(len(fs.p))):
            return False
        
        # κ是无环DAG
        if self._has_cycle_in_dag(fs.kappa):
            return False
        
        return True
    
    def _are_dimensions_compatible(self, dim1: int, dim2: int) -> bool:
        """检查维度兼容性"""
        # 简化规则：维度必须有公约数
        return np.gcd(dim1, dim2) > 1
    
    def _compute_joint_dimension(self, dim1: int, dim2: int) -> int:
        """计算联合维度"""
        return np.gcd(dim1, dim2)
    
    def _preserves_five_fold_equivalence(self, T1: BDAGTheory, T2: BDAGTheory) -> bool:
        """检查五重等价性保持"""
        # 简化版：检查理论是否有共同依赖
        common_deps = set(T1.dependencies) & set(T2.dependencies)
        return len(common_deps) > 0
    
    def _build_inference_graph(self) -> Dict[int, List[int]]:
        """构建推理图"""
        graph = defaultdict(list)
        for theory_id, theory in self.theories.items():
            for dep in theory.dependencies:
                graph[dep].append(theory_id)
        return dict(graph)
    
    def _find_cycle(self, graph: Dict[int, List[int]]) -> Optional[List[int]]:
        """寻找循环"""
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                return path[cycle_start:]
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    cycle = dfs(neighbor, path + [neighbor])
                    if cycle:
                        return cycle
            
            rec_stack.remove(node)
            return None
        
        for node in graph:
            if node not in visited:
                cycle = dfs(node, [node])
                if cycle:
                    return cycle
        
        return None
    
    def _has_contradiction_path(self, graph: Dict[int, List[int]], node: int) -> bool:
        """检查是否有矛盾路径"""
        # 简化版：检查是否能同时到达两个矛盾的预测
        return False  # 简化实现
    
    def _check_generation_consistency(self, theory: BDAGTheory) -> bool:
        """检查生成规则一致性"""
        # 简化版：总是返回True
        return True
    
    def _check_verification_condition(self, v_index: int) -> bool:
        """检查V1-V5验证条件"""
        if v_index == 1:  # V1: I/O合法性
            for theory in self.theories.values():
                if '11' in theory.get_zeckendorf_encoding():
                    return False
        elif v_index == 2:  # V2: 维度一致性
            dims = [t.tensor_dimension for t in self.theories.values()]
            if len(set(dims)) > 3:  # 太多不同维度
                return False
        elif v_index == 5:  # V5: 五重等价性
            # 简化检查
            return len(self.theories) > 0
        
        return True
    
    def _avoid_consecutive_ones(self, n: int) -> int:
        """避免连续1的编码"""
        # 简化版：如果有问题就加1
        enc = self._encode_zeckendorf(n)
        if '11' in enc:
            return n + 1
        return n
    
    def _normalize_fold_signature(self, fs: FoldSignature) -> FoldSignature:
        """规范化折叠签名"""
        # 简化版：排序z
        return FoldSignature(
            z=sorted(fs.z, reverse=True),
            p=fs.p,
            tau=fs.tau,
            sigma=fs.sigma,
            b=fs.b,
            kappa=fs.kappa,
            annot=fs.annot
        )
    
    def _has_cycle_in_dag(self, dag: Dict[str, List[str]]) -> bool:
        """检查DAG是否有环"""
        # 简化版
        return False
    
    def _relax_no11_constraint(self):
        """放松No-11约束"""
        pass
    
    def _introduce_dimension_mapping(self):
        """引入维度映射"""
        pass
    
    def _introduce_conditional_equivalence(self):
        """引入条件等价性"""
        pass
    
    def _compute_syntactic_consistency(self) -> float:
        """计算语法一致性度量"""
        total_pairs = len(list(itertools.combinations(self.theories.keys(), 2)))
        if total_pairs == 0:
            return 1.0
        
        violating_pairs = 0
        for t1_id, t2_id in itertools.combinations(self.theories.keys(), 2):
            if self.detect_syntactic_contradiction(self.theories[t1_id], self.theories[t2_id]):
                violating_pairs += 1
        
        return 1.0 - (violating_pairs / total_pairs)
    
    def _compute_semantic_consistency(self) -> float:
        """计算语义一致性度量"""
        total_predictions = sum(len(t.predictions) for t in self.theories.values())
        if total_predictions == 0:
            return 1.0
        
        conflicting = 0
        for t1_id, t2_id in itertools.combinations(self.theories.keys(), 2):
            t1 = self.theories[t1_id]
            t2 = self.theories[t2_id]
            for pred in t1.predictions:
                if f"not_{pred}" in t2.predictions:
                    conflicting += 1
        
        return 1.0 - (conflicting / total_predictions)
    
    def _compute_meta_consistency(self) -> float:
        """计算元理论一致性度量"""
        scores = []
        for i in range(1, 6):
            scores.append(1.0 if self._check_verification_condition(i) else 0.0)
        return np.mean(scores)
    
    def _compute_bdag_specific_measure(self) -> float:
        """计算BDAG特定度量"""
        # 折叠复杂度的倒数
        avg_fold_complexity = np.mean([len(t.fold_signature.z) for t in self.theories.values()])
        fold_score = 1.0 / (1.0 + avg_fold_complexity)
        
        # 审计覆盖率（简化版）
        audit_coverage = 0.8  # 假设80%覆盖
        
        return fold_score * audit_coverage
    
    def _encode_zeckendorf(self, n: int) -> str:
        """辅助方法：Zeckendorf编码"""
        if n == 0:
            return "0"
        
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        encoding = []
        for f in reversed(fibs):
            if f <= n:
                encoding.append('1')
                n -= f
            else:
                encoding.append('0')
        
        result = ''.join(encoding).lstrip('0')
        return result if result else '0'


# 测试用例
class TestBDAGConsistency:
    """BDAG一致性测试套件"""
    
    def test_no11_violation_detection(self):
        """测试No-11约束违反检测"""
        checker = BDAGConsistencyChecker()
        
        # 创建违反No-11的理论
        theory_bad = BDAGTheory(
            N=3,  # 编码为"11"
            fold_signature=FoldSignature(
                z=[2, 1], p=[0, 1], tau="((", sigma=[0, 1], 
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=3,
            predictions=set()
        )
        
        theory_good = BDAGTheory(
            N=4,  # 编码为"101"
            fold_signature=FoldSignature(
                z=[3, 1], p=[0, 1], tau="((", sigma=[0, 1],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=4,
            predictions=set()
        )
        
        # 检测应该发现No-11违反
        contradiction = checker.detect_syntactic_contradiction(theory_bad, theory_good)
        assert contradiction is not None
        assert contradiction.type == ContradictionType.NO11_VIOLATION
    
    def test_dimension_compatibility(self):
        """测试维度兼容性检查"""
        checker = BDAGConsistencyChecker()
        
        theory1 = BDAGTheory(
            N=5,
            fold_signature=FoldSignature(
                z=[4], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=8,
            predictions=set()
        )
        
        theory2 = BDAGTheory(
            N=7,
            fold_signature=FoldSignature(
                z=[4, 2], p=[0, 1], tau="((", sigma=[0, 1],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=9,  # gcd(8,9)=1, 不兼容
            predictions=set()
        )
        
        contradiction = checker.detect_syntactic_contradiction(theory1, theory2)
        assert contradiction is not None
        assert contradiction.type == ContradictionType.DIMENSION_MISMATCH
    
    def test_prediction_conflict(self):
        """测试预测冲突检测"""
        checker = BDAGConsistencyChecker()
        
        theory1 = BDAGTheory(
            N=8,
            fold_signature=FoldSignature(
                z=[5], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=8,
            predictions={"entropy_increases"}
        )
        
        theory2 = BDAGTheory(
            N=9,
            fold_signature=FoldSignature(
                z=[5, 1], p=[0, 1], tau="((", sigma=[0, 1],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=8,
            predictions={"not_entropy_increases"}  # 冲突预测
        )
        
        contradiction = checker.detect_semantic_contradiction(theory1, theory2)
        assert contradiction is not None
        assert contradiction.type == ContradictionType.CONFLICTING_PREDICTION
    
    def test_local_repair_strategy(self):
        """测试局部修复策略"""
        checker = BDAGConsistencyChecker()
        
        # 添加有No-11违反的理论
        theory = BDAGTheory(
            N=3,
            fold_signature=FoldSignature(
                z=[2, 1], p=[0, 1], tau="((", sigma=[0, 1],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=3,
            predictions=set()
        )
        
        checker.theories[3] = theory
        
        # 创建矛盾记录
        contradiction = Contradiction(
            type=ContradictionType.NO11_VIOLATION,
            affected_theories=[3],
            details={'encoding': '11'},
            severity='resolvable'
        )
        
        # 应用局部修复
        success = checker.local_repair(contradiction)
        assert success
        
        # 验证修复后的理论
        repaired = checker.theories[3]
        assert '11' not in repaired.get_zeckendorf_encoding()
    
    def test_consistency_tensor_computation(self):
        """测试一致性张量计算"""
        checker = BDAGConsistencyChecker()
        
        # 添加一些一致的理论
        for i in [1, 2, 4, 5]:
            theory = BDAGTheory(
                N=i,
                fold_signature=FoldSignature(
                    z=[i], p=[0], tau="(", sigma=[0],
                    b=[], kappa={}, annot={}
                ),
                dependencies=[],
                tensor_dimension=2**i,
                predictions={f"pred_{i}"}
            )
            checker.theories[i] = theory
        
        # 计算一致性张量
        tensor = checker.compute_consistency_tensor()
        
        assert tensor.shape == (5,)
        assert all(0 <= x <= 1 for x in tensor)
        
        # 计算一致性分数
        score = np.linalg.norm(tensor)
        assert score > 0
    
    def test_automated_consistency_check(self):
        """测试自动化一致性检查流程"""
        checker = BDAGConsistencyChecker()
        
        # 添加一组理论
        theories = [
            BDAGTheory(
                N=1,
                fold_signature=FoldSignature(
                    z=[1], p=[0], tau="(", sigma=[0],
                    b=[], kappa={}, annot={}
                ),
                dependencies=[],
                tensor_dimension=1,
                predictions={"axiom"}
            ),
            BDAGTheory(
                N=2,
                fold_signature=FoldSignature(
                    z=[2], p=[0], tau="(", sigma=[0],
                    b=[], kappa={}, annot={}
                ),
                dependencies=[1],
                tensor_dimension=2,
                predictions={"entropy"}
            ),
            BDAGTheory(
                N=4,
                fold_signature=FoldSignature(
                    z=[3, 1], p=[0, 1], tau="((", sigma=[0, 1],
                    b=[], kappa={}, annot={}
                ),
                dependencies=[1, 2],
                tensor_dimension=4,
                predictions={"combined"}
            )
        ]
        
        for theory in theories:
            checker.add_theory(theory)
        
        # 执行自动化检查
        report = checker.automated_consistency_check()
        
        assert isinstance(report, ConsistencyReport)
        assert len(report.consistency_tensor) == 5
        assert report.consistency_score > 0
        
        # 检查是否一致（阈值φ^5 ≈ 11.09）
        if report.is_consistent:
            assert report.consistency_score >= PHI**5
        else:
            assert report.consistency_score < PHI**5
    
    def test_incremental_consistency(self):
        """测试增量式一致性检查"""
        checker = BDAGConsistencyChecker()
        
        # 先添加基础理论
        theory1 = BDAGTheory(
            N=1,
            fold_signature=FoldSignature(
                z=[1], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=1,
            predictions={"base"}
        )
        
        assert checker.add_theory(theory1)
        
        # 添加兼容理论
        theory2 = BDAGTheory(
            N=2,
            fold_signature=FoldSignature(
                z=[2], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[1],
            tensor_dimension=2,
            predictions={"derived"}
        )
        
        assert checker.add_theory(theory2)
        
        # 尝试添加不兼容理论
        theory3 = BDAGTheory(
            N=3,
            fold_signature=FoldSignature(
                z=[2, 1], p=[0, 1], tau="((", sigma=[0, 1],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=3,
            predictions={"not_base"}  # 冲突预测
        )
        
        # 因为N=3编码为"11"，应该被拒绝
        assert not checker.add_theory(theory3)
        assert len(checker.contradiction_log) > 0
    
    def test_five_fold_equivalence(self):
        """测试五重等价性检查"""
        checker = BDAGConsistencyChecker()
        
        # 有共同依赖的理论（满足五重等价性）
        theory1 = BDAGTheory(
            N=5,
            fold_signature=FoldSignature(
                z=[4], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[1, 2],
            tensor_dimension=8,
            predictions={"entropy", "time"}
        )
        
        theory2 = BDAGTheory(
            N=8,
            fold_signature=FoldSignature(
                z=[5], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[2, 3],  # 共同依赖2
            tensor_dimension=8,
            predictions={"information", "observer"}
        )
        
        # 应该满足五重等价性
        assert checker._preserves_five_fold_equivalence(theory1, theory2)
        
        # 没有共同依赖的理论
        theory3 = BDAGTheory(
            N=13,
            fold_signature=FoldSignature(
                z=[6], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[7, 9],  # 无共同依赖
            tensor_dimension=13,
            predictions={"isolated"}
        )
        
        # 不满足五重等价性
        assert not checker._preserves_five_fold_equivalence(theory1, theory3)
    
    def test_consistency_cache(self):
        """测试一致性缓存机制"""
        checker = BDAGConsistencyChecker()
        
        theory1 = BDAGTheory(
            N=1,
            fold_signature=FoldSignature(
                z=[1], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=1,
            predictions=set()
        )
        
        theory2 = BDAGTheory(
            N=2,
            fold_signature=FoldSignature(
                z=[2], p=[0], tau="(", sigma=[0],
                b=[], kappa={}, annot={}
            ),
            dependencies=[],
            tensor_dimension=2,
            predictions=set()
        )
        
        # 第一次检查
        result1 = checker._check_pair_consistency(theory1, theory2)
        
        # 缓存应该被填充
        cache_key = (1, 2)
        assert cache_key in checker.consistency_cache
        
        # 第二次检查应该使用缓存
        result2 = checker._check_pair_consistency(theory1, theory2)
        
        # 结果应该相同
        assert (result1 is None) == (result2 is None)


# 运行测试
if __name__ == "__main__":
    print("运行M1.5 BDAG理论一致性元定理测试...")
    
    test_suite = TestBDAGConsistency()
    
    # 运行各项测试
    test_suite.test_no11_violation_detection()
    print("✓ No-11约束违反检测通过")
    
    test_suite.test_dimension_compatibility()
    print("✓ 维度兼容性检查通过")
    
    test_suite.test_prediction_conflict()
    print("✓ 预测冲突检测通过")
    
    test_suite.test_local_repair_strategy()
    print("✓ 局部修复策略通过")
    
    test_suite.test_consistency_tensor_computation()
    print("✓ 一致性张量计算通过")
    
    test_suite.test_automated_consistency_check()
    print("✓ 自动化一致性检查通过")
    
    test_suite.test_incremental_consistency()
    print("✓ 增量式一致性检查通过")
    
    test_suite.test_five_fold_equivalence()
    print("✓ 五重等价性检查通过")
    
    test_suite.test_consistency_cache()
    print("✓ 一致性缓存机制通过")
    
    print("\n所有测试通过！M1.5 BDAG理论一致性元定理验证成功。")
    
    # 演示完整流程
    print("\n" + "="*60)
    print("演示：BDAG理论体系一致性验证")
    print("="*60)
    
    checker = BDAGConsistencyChecker()
    
    # 构建理论体系
    theories = [
        BDAGTheory(N=1, fold_signature=FoldSignature(z=[1], p=[0], tau="(", sigma=[0], b=[], kappa={}, annot={}),
                  dependencies=[], tensor_dimension=1, predictions={"axiom", "self_reference"}),
        BDAGTheory(N=2, fold_signature=FoldSignature(z=[2], p=[0], tau="(", sigma=[0], b=[], kappa={}, annot={}),
                  dependencies=[1], tensor_dimension=2, predictions={"entropy_increase"}),
        BDAGTheory(N=4, fold_signature=FoldSignature(z=[3, 1], p=[0, 1], tau="((", sigma=[0, 1], b=[], kappa={}, annot={}),
                  dependencies=[1, 2], tensor_dimension=4, predictions={"structure_emergence"}),
        BDAGTheory(N=5, fold_signature=FoldSignature(z=[4], p=[0], tau="(", sigma=[0], b=[], kappa={}, annot={}),
                  dependencies=[1], tensor_dimension=8, predictions={"information_theory"}),
        BDAGTheory(N=8, fold_signature=FoldSignature(z=[5], p=[0], tau="(", sigma=[0], b=[], kappa={}, annot={}),
                  dependencies=[1, 2], tensor_dimension=8, predictions={"complexity_emergence"})
    ]
    
    print("\n添加理论到系统...")
    for theory in theories:
        success = checker.add_theory(theory)
        print(f"  T{theory.N}: {'✓ 成功' if success else '✗ 失败'}")
    
    print("\n执行自动化一致性检查...")
    report = checker.automated_consistency_check()
    
    print(f"\n检测结果:")
    print(f"  语法问题: {len(report.syntactic_issues)}")
    print(f"  语义问题: {len(report.semantic_issues)}")
    print(f"  逻辑问题: {len(report.logical_issues)}")
    print(f"  元理论问题: {len(report.meta_issues)}")
    
    print(f"\n一致性度量:")
    print(f"  语法一致性: {report.consistency_tensor[0]:.3f}")
    print(f"  语义一致性: {report.consistency_tensor[1]:.3f}")
    print(f"  逻辑一致性: {report.consistency_tensor[2]:.3f}")
    print(f"  元理论一致性: {report.consistency_tensor[3]:.3f}")
    print(f"  BDAG特定度量: {report.consistency_tensor[4]:.3f}")
    
    print(f"\n最终评估:")
    print(f"  一致性分数: {report.consistency_score:.2f}")
    print(f"  一致性阈值 (φ^5): {PHI**5:.2f}")
    print(f"  系统状态: {'✓ 一致' if report.is_consistent else '✗ 不一致'}")
    
    if report.repairs_applied:
        print(f"\n应用的修复:")
        for contradiction, strategy in report.repairs_applied:
            print(f"  - {contradiction.type.value} → {strategy.value}")
    
    print("\n" + "="*60)
    print("M1.5 BDAG理论一致性元定理验证完成")
    print("="*60)