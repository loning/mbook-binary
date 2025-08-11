# T0 理论体系符号修正清单

## 生成日期：2025-08-11

基于符号使用分析和新建立的符号规范，以下文件需要进行符号标准化修正。

## 优先级说明
- 🔴 **高优先级**：核心符号冲突，影响理解
- 🟡 **中优先级**：局部不一致，需要统一
- 🟢 **低优先级**：格式规范化，提高可读性

## 1. 需要立即修正的文件（🔴 高优先级）

### T0-1-binary-state-space-foundation.md
**问题**：
- 使用S(B)表示熵，应改为H(B)
- 符号σ的定义需要更明确的说明
**修正建议**：
```diff
- S(B) = -∑ᵢ p(bᵢ)log₂(p(bᵢ)) + λ·violations(B)
+ H(B) = -∑ᵢ p(bᵢ)log₂(p(bᵢ)) + λ·violations(B)
```

### T0-5-entropy-flow-conservation-theory.md
**问题**：
- S₁表示系统，容易与熵函数S混淆
- 流率符号Φ需要统一定义
**修正建议**：
- 明确S₁为系统标识，不是熵
- 在文档开头添加符号定义节

### T0-10-entropy-capacity-scaling-theory.md
**问题**：
- 多个希腊字母(α, β, γ, δ, ε, ξ, ν)未充分定义
- 缩放指数α的多种表达式需要统一
**修正建议**：
- 添加完整的符号定义表
- 统一α的主要定义，其他作为特殊情况

## 2. 需要统一的符号使用（🟡 中优先级）

### 熵函数标记不一致
**涉及文件**：
- T0-1: 使用S(B)
- T0-2至T0-20: 使用H(·)

**统一方案**：
全部改为H(·)表示熵，S专门表示系统

### 时间量子符号
**涉及文件**：
- T0-0: τ₀定义为最小自指时间
- T0-13: τ₀用作厚度量子
- T0-16: τ₀在能量公式中

**统一方案**：
- 主定义保持在T0-0
- 其他文件明确引用"T0-0中定义的τ₀"

### Zeckendorf编码函数
**涉及文件**：
- 部分文件使用Z(n)
- 部分文件使用[n]_φ
- 部分文件使用Z_φ(n)

**统一方案**：
统一使用Z(n)作为标准记法

## 3. 格式规范化需求（🟢 低优先级）

### 下标格式不一致
**问题示例**：
- 混用S_0, S₀
- 混用F_n, Fₙ
- 混用H_1, H₁

**修正方案**：
- 数字下标统一使用Unicode下标字符：₀₁₂₃₄₅₆₇₈₉
- 字母下标使用TeX格式：_i, _j, _k

### 函数参数格式
**问题示例**：
- H(S,t) vs H(S(t)) vs H_t(S)

**修正方案**：
- 时间依赖：H(S, t)
- 时间演化：H(S(t))
- 时刻快照：H_t(S)

## 4. 具体文件修正清单

### 批次1（第一周完成）
- [ ] T0-1-binary-state-space-foundation.md - S(B)→H(B)
- [ ] T0-5-entropy-flow-conservation-theory.md - 添加符号定义
- [ ] T0-10-entropy-capacity-scaling-theory.md - 规范希腊字母使用

### 批次2（第二周完成）
- [ ] T0-6-system-component-interaction-theory.md - 统一耦合参数ε
- [ ] T0-8-minimal-information-principle-theory.md - 规范η的定义
- [ ] T0-11-recursive-depth-hierarchy.md - 统一深度符号d vs n

### 批次3（第三周完成）
- [ ] T0-13-system-boundaries.md - 澄清τ₀的使用
- [ ] T0-14-discrete-continuous-transition.md - 规范极限符号
- [ ] T0-15-spatial-dimension-emergence.md - 统一维度符号d

### 批次4（第四周完成）
- [ ] T0-16-information-energy-equivalence.md - 能量符号规范化
- [ ] T0-17-information-entropy-zeckendorf.md - 统一编码函数记法
- [ ] T0-18-quantum-state-emergence.md - 量子态符号标准化
- [ ] T0-19-observation-collapse.md - 测量算子符号统一

### 批次5（第五周完成）
- [ ] T0-20-zeckendorf-metric-space-foundation.md - 度量符号规范
- [ ] 所有formal文件 - 确保与主文件一致
- [ ] 测试文件 - 更新以匹配新符号

## 5. 自动化修正脚本

```python
# symbol_correction.py
import re
import os

def correct_entropy_symbol(content):
    """将S(B)形式的熵改为H(B)"""
    # 匹配S(...)形式的熵表达式
    pattern = r'S\(([^)]+)\)\s*=\s*-∑'
    replacement = r'H(\1) = -∑'
    return re.sub(pattern, replacement, content)

def correct_subscripts(content):
    """统一下标格式"""
    subscript_map = {
        '_0': '₀', '_1': '₁', '_2': '₂', '_3': '₃',
        '_4': '₄', '_5': '₅', '_6': '₆', '_7': '₇',
        '_8': '₈', '_9': '₉'
    }
    for old, new in subscript_map.items():
        content = content.replace(old, new)
    return content

def add_symbol_definitions(content, theory_name):
    """在文档开头添加符号定义"""
    if "## 符号定义" not in content:
        definitions = generate_symbol_definitions(theory_name)
        # 在Abstract后插入
        insert_pos = content.find("## 1.")
        if insert_pos > 0:
            content = content[:insert_pos] + definitions + "\n" + content[insert_pos:]
    return content

def process_file(filepath):
    """处理单个文件"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    content = correct_entropy_symbol(content)
    content = correct_subscripts(content)
    content = add_symbol_definitions(content, os.path.basename(filepath))
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# 运行修正
if __name__ == "__main__":
    files_to_correct = [
        "T0-1-binary-state-space-foundation.md",
        # ... 其他文件
    ]
    
    for filename in files_to_correct:
        filepath = f"/Users/cookie/mbook-binary/src/binaryuniverse/{filename}"
        if process_file(filepath):
            print(f"✓ Corrected: {filename}")
        else:
            print(f"- No changes: {filename}")
```

## 6. 验证检查清单

修正完成后，每个文件必须通过以下检查：

### 自动检查
- [ ] 运行 `check_symbols.py` 无错误
- [ ] 运行 `detect_conflicts.py` 无冲突
- [ ] LaTeX编译无警告

### 人工检查
- [ ] 符号定义完整
- [ ] 交叉引用正确
- [ ] 格式统一美观
- [ ] 物理含义清晰

## 7. 进度跟踪

| 理论编号 | 文件名 | 状态 | 负责人 | 完成日期 |
|---------|--------|------|--------|----------|
| T0-0 | time-emergence-foundation | ✅ 已规范 | - | - |
| T0-1 | binary-state-space | 🔄 进行中 | - | - |
| T0-2 | fundamental-entropy-bucket | ⏳ 待处理 | - | - |
| ... | ... | ... | ... | ... |

## 8. 常见问题修正指南

### Q: 如何处理历史遗留的S(B)熵表示？
A: 全部改为H(B)，并在改动处添加注释说明

### Q: 局部变量与全局符号冲突怎么办？
A: 局部变量改用不同符号或添加明确下标

### Q: formal文件与主文件不一致？
A: 以主文件为准，formal文件随后更新

### Q: 是否需要保留原始版本？
A: 在git中保留，文档中不需要显示修改痕迹

## 9. 时间表

- **第1周**：高优先级修正
- **第2-4周**：中优先级统一
- **第5周**：低优先级格式化
- **第6周**：最终验证和文档更新

## 10. 反馈机制

发现新的符号问题请：
1. 在此文档中添加
2. 标注优先级
3. 提出修正建议
4. 通知维护团队

---

**最后更新**：2025-08-11
**下次审查**：2025-09-11

∎