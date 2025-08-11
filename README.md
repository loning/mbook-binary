# 二进制宇宙理论文档项目

这是一个基于mdBook构建的二进制宇宙理论完整文档系统，包含KaTeX数学公式渲染支持。项目已完成T0基础理论层的全面构建，通过492个单元测试验证了从时间涌现到纠错码理论的完整理论体系，为T1-T33高层理论提供了坚实的数学基础。

## 项目特性

- **严格数学验证**: 492个单元测试全部通过验证，确保理论预言的计算可验证性
- **Zeckendorf编码体系**: 完整实现φ-编码系统，No-11约束确保理论一致性
- **最小完备性原则**: 每个理论都遵循Definition→Theorem→Proof的严格学术结构
- **形式化证明系统**: 包含完整的formal目录，提供T0-T33系列的形式化证明文件
- **数学公式支持**: 使用mdbook-katex渲染LaTeX数学公式
- **自动化部署**: 通过GitHub Actions自动构建并部署到GitHub Pages
- **层次化导航**: 完整的SUMMARY.md目录结构，支持章节和小节导航

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