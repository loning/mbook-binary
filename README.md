# 二进制宇宙理论文档项目

这是一个基于mdBook构建的二进制宇宙理论完整文档系统，包含KaTeX数学公式渲染支持。

## 项目特性

- **完整理论体系**: 包含505个markdown理论文档，涵盖35个主要章节
- **形式化证明系统**: 包含完整的formal目录，提供247个形式化证明文件
- **数学公式支持**: 使用mdbook-katex渲染LaTeX数学公式
- **自动化部署**: 通过GitHub Actions自动构建并部署到GitHub Pages
- **层次化导航**: 完整的SUMMARY.md目录结构，支持章节和小节导航

## 理论内容概览

### 核心理论体系
- **第0章**: 基础理论体系 (T0.1-T0.10)
- **第1章**: 唯一公理 (A1五重等价性)
- **第2章**: 信息编码理论体系 (定义D1.1-D1.9, 引理L1.1-L1.8, 定理T1.1-T2.12)
- **第3-35章**: 量子现象、数学结构、信息理论、计算复杂度、宇宙学应用等

### 形式化证明系统
- **公理系统形式化**: A1公理的严格形式化定义
- **基础定义形式化**: D1.1-D1.9的形式化表述
- **定理形式化**: T0-T33系列的完整形式化证明
- **推论形式化**: C1-C21系列推论的形式化
- **命题形式化**: P1-P10基础命题的形式化
- **元定理形式化**: M1-M3元数学定理

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