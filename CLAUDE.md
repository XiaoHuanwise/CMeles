# CMeles

## 项目概述

CMeles 是一个基于 C++ 的可压缩流动求解器，采用间断 Galerkin（Discontinuous Galerkin, DG）方法进行空间离散化。项目面向学术研究，兼顾计算性能与代码可扩展性。

## 技术栈

| 组件                 | 用途                                                         |
| -------------------- | ------------------------------------------------------------ |
| **OCCA** (`libocca`) | 异构计算后端，支持 CPU 和 GPU，核函数统一用 OKL 编写         |
| **Eigen**            | 数据准备阶段的矩阵/向量运算（如 Vandermonde 矩阵、质量矩阵） |
| **HighFive**         | HDF5 文件 I/O，用于结果输出                                  |
| **toml++**           | TOML 格式配置文件解析                                        |

## 项目结构

- `CMeles.cpp` — 程序入口
- `src/` — 源码目录
- `tests/` — 测试用例（环境验证测试直接放在该目录下，其他测试用例按功能放入对应子目录）
- `docs/` — 项目文档（中文）（安装指南、使用说明、代码结构、算法原理等）

## 代码规范

### 编码风格

- 使用 `.clang-format` 定义的代码风格，提交前应格式化代码
- 使用 C++17 或更新标准
- 优先使用现代 C++ 特性（`constexpr`、`std::unique_ptr`/`std::shared_ptr`、lambda 表达式、结构化绑定等）

### 注释

- 使用 LLVM 注释规范，注释语言为英文
- 作为学术开发代码，关键算法步骤和物理含义处应有适当的行内注释

### 设计原则

- **静态多态优先**：使用 CRTP 等技术实现编译期多态，避免不必要的虚函数，减少运行时开销
- **RAII**：严格遵循 RAII 原则管理资源（内存、文件句柄、设备上下文等），确保异常安全
- **设计模式**：根据场景合理使用工厂模式、单例模式等提升可维护性

## OCCA 核函数开发

所有计算核函数定义在 `.okl` 文件中。编写核函数时需根据目标平台准备两套逻辑：

- **CPU 路径**：不使用 `block`/`@outer` 共享内存
- **GPU 路径**：使用 `block`/`@outer` 共享内存以利用 GPU 的片上存储

通过宏定义切换两套代码路径。

## 数据输出

使用 HighFive 库将计算结果写入 HDF5 文件，数据采用 SoA（Structure of Arrays）布局直接保存积分点数据。

## Git 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，commit message 使用英文。格式：

```
<type>(<scope>): <description>

[optional body]
```

常用类型：`feat`、`fix`、`docs`、`refactor`、`test`、`chore`、`perf`。

## 文档

项目文档使用 Markdown 格式，放置在 `docs/` 目录下。文档应涵盖安装方法、使用指南、代码架构和核心算法原理。
