# 二进制宇宙理论文档项目

这是一个基于mdBook构建的二进制宇宙理论完整文档系统，包含KaTeX数学公式渲染支持。项目已完成T0基础理论层的全面构建，通过108个单元测试验证了从时间涌现到观察坍缩的完整理论体系，为T1-T33高层理论提供了坚实的数学基础。

## 项目特性

- **完整T0基础理论**: 20个核心T0理论(T0-0至T0-19)，涵盖时间涌现、空间维度、信息-能量等价、熵编码、量子态涌现和观察坍缩
- **严格数学验证**: 108个单元测试全部通过验证，确保理论预言的计算可验证性
- **Zeckendorf编码体系**: 完整实现φ-编码系统，No-11约束确保理论一致性
- **最小完备性原则**: 每个理论都遵循Definition→Theorem→Proof的严格学术结构
- **形式化证明系统**: 包含完整的formal目录，提供T0-T33系列的形式化证明文件
- **数学公式支持**: 使用mdbook-katex渲染LaTeX数学公式
- **自动化部署**: 通过GitHub Actions自动构建并部署到GitHub Pages
- **层次化导航**: 完整的SUMMARY.md目录结构，支持章节和小节导航

## 理论内容概览

### 核心理论体系

#### T0基础理论层 (完整构建)
- **T0-0**: 时间涌现基础理论 - 从A1公理推导时间必然性
- **T0-1至T0-10**: 原始基础理论体系 - 二进制状态空间、熵流守恒等
- **T0-11**: 递归深度层次理论 - 递归深度量化和Fibonacci层次结构
- **T0-12**: 观察者涌现理论 - 观察者分化必然性和信息成本模型
- **T0-13**: 系统边界理论 - 边界涌现、量化和信息流调节
- **T0-14**: 离散-连续过渡理论 - 连续性从φ收敛的数学涌现
- **T0-15**: 空间维度涌现理论 - 从φ-正交性推导3+1维时空结构
- **T0-16**: 信息-能量等价理论 - 能量作为信息处理速率的涌现
- **T0-17**: 信息熵Zeckendorf编码理论 - 熵的φ-量化表示和Fibonacci增长
- **T0-18**: 量子态No-11约束涌现理论 - 量子叠加、坍缩和Born规则的推导
- **T0-19**: 观察坍缩信息过程理论 - 观察导致波函数坍缩的信息理论机制

#### 高层理论体系
- **第1章**: 唯一公理 (A1五重等价性)
- **第2章**: 信息编码理论体系 (定义D1.1-D1.9, 引理L1.1-L1.8, 定理T1.1-T2.12)
- **第3-35章**: 量子现象、数学结构、信息理论、计算复杂度、宇宙学应用等

### 形式化证明系统
- **T0理论完整形式化**: 20个T0理论的严格Definition→Theorem→Proof结构
- **公理系统形式化**: A1公理的严格形式化定义和五重等价性
- **基础定义形式化**: D1.1-D1.9的形式化表述
- **高层定理形式化**: T1-T33系列的完整形式化证明
- **推论形式化**: C1-C21系列推论的形式化
- **命题形式化**: P1-P10基础命题的形式化
- **元定理形式化**: M1-M3元数学定理

### T0理论核心创新
1. **时间从自指悖论涌现** - 不是假设时间存在，而是从A1公理推导时间必然性
2. **3+1维时空必然性** - 从φ-正交性约束严格推导为什么空间恰好是3维
3. **信息-能量统一** - 能量是信息处理速率，E=(dI/dt)×ℏ_φ，统一质能关系
4. **量子力学完全涌现** - 量子叠加、坍缩、Born规则都从No-11约束必然推导
5. **观察坍缩机制** - 观察者与量子系统的信息交换导致坍缩的精确机制
6. **熵的φ-量化** - 信息熵遵循Fibonacci量子化增长模式
7. **所有结构遵循Fibonacci量化** - 递归深度、时间间隔、能量等级都遵循黄金比例

## 本地开发

### 环境要求

- Rust 工具链
- mdbook
- mdbook-katex

### 安装步骤

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 mdbook 和 mdbook-katex
cargo install mdbook mdbook-katex
```

### 构建项目

```bash
# 构建文档
mdbook build

# 本地服务器（支持热重载）
mdbook serve
```

网站将在 http://localhost:3000 可访问

## GitHub Pages Deployment

1. Push your changes to the `main` branch
2. GitHub Actions will automatically build and deploy to GitHub Pages
3. Enable GitHub Pages in repository settings:
   - Go to Settings → Pages
   - Source: GitHub Actions

## Math Support

This project supports LaTeX mathematics via mdbook-katex:

### Inline Math
Use `$...$` for inline equations: $E = mc^2$

### Display Math
Use `$$...$$` for display equations:

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

## Project Structure

```
├── .github/workflows/mdbook.yml  # GitHub Actions workflow
├── src/
│   ├── SUMMARY.md               # Table of contents
│   └── chapter_1.md             # Example chapter
├── book.toml                    # mdBook configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

Edit `book.toml` to customize your book settings. The KaTeX preprocessor is already configured.