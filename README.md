# Documentation Project

This is an mdBook project with KaTeX support for rendering mathematical formulas.

## Features

- **mdBook**: Static site generator for documentation
- **mdbook-katex**: LaTeX math rendering support
- **GitHub Pages**: Automatic deployment via GitHub Actions

## Local Development

### Prerequisites

- Rust toolchain
- mdbook
- mdbook-katex

### Installation

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install mdbook and mdbook-katex
cargo install mdbook mdbook-katex
```

### Building

```bash
# Build the book
mdbook build

# Serve locally with auto-reload
mdbook serve
```

The site will be available at http://localhost:3000

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