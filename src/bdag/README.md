# Binary Universe DAG (BDAG) 规范文档

本目录包含二进制宇宙理论的有向无环图（DAG）文件命名和组织规范，用于将理论原子化为单一张量操作，确保理论结构的清晰性、可追溯性和可计算性。

## 目录结构

```
/src/bdag/
├── README.md                    # 本文件：总体介绍
├── NAMING_SPECIFICATION.md     # 文件命名规范
├── TENSOR_OPERATIONS.md         # 张量操作类型定义
├── LAYER_ARCHITECTURE.md       # 层级架构设计
├── PARSING_TOOLS.md            # 解析工具和实现
├── MIGRATION_GUIDE.md          # 从现有文件迁移指南
├── VALIDATION_RULES.md         # 验证规则和检查
├── examples/                   # 命名示例
│   ├── layer_0_examples.md     # 第0层命名示例
│   ├── layer_1_examples.md     # 第1层命名示例
│   ├── layer_2_examples.md     # 第2层命名示例
│   └── layer_3_examples.md     # 第3层命名示例
└── tools/                      # 相关工具
    ├── parser.py               # 文件名解析器
    ├── validator.py            # DAG验证器
    ├── visualizer.py           # DAG可视化工具
    └── migrator.py             # 文件迁移工具
```

## 核心原则

### 1. 原子性原则
每个文件只包含一个张量操作，不能混合多个概念或操作。

### 2. DAG结构原则
文件名必须体现有向无环图的结构关系，包括层级、依赖和顺序。

### 3. 可解析原则
文件名必须采用标准化格式，支持程序自动解析和处理。

### 4. 语义清晰原则
文件名必须包含足够的语义信息，使人类和程序都能理解其含义。

### 5. 数学严谨原则
每个张量操作必须有明确的数学定义和验证规则。

## 快速开始

1. 阅读 [NAMING_SPECIFICATION.md](./NAMING_SPECIFICATION.md) 了解命名规范
2. 查看 [examples/](./examples/) 目录下的具体示例
3. 使用 [tools/parser.py](./tools/parser.py) 验证文件名格式
4. 参考 [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) 迁移现有文件

## 版本信息

- 规范版本：v1.0.0
- 最后更新：2024年8月
- 维护者：Binary Universe Theory Project

## 许可证

本规范遵循项目整体许可证。