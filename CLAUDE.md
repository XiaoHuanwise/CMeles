每次回答都要称呼我为 "Helldiver"。

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
- `docs/` — 项目文档（中文内容，英文文件名）（安装指南、使用说明、代码结构、算法原理等）

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

所有计算核函数定义在 `.okl` 文件中。所有后端（GPU CUDA/HIP、CPU OpenMP/Serial）统一使用 `@tile` 对 elem 循环标注，无需宏定义切换不同代码路径。

- **所有路径统一**：使用 `@tile(TILE_SIZE, @outer, @inner)` 对 elem 循环分块。`@tile` 自动拆分为外层 `@outer`（GPU 映射到 block、OpenMP 映射到 `#pragma omp parallel for`）和内层 `@inner`（GPU 映射到 thread、OpenMP/Serial 退化为普通串行循环），elem 内全部串行。当前不考虑 `@shared` 共享内存。
- **SIMD 向量化**：不依赖 OKL 的 `@inner` 标注，而是由编译器在 `-O3 -march=native` 下自动对最内层连续内存访问循环进行向量化。
- **平台间差异**：仅 `TILE_SIZE` 参数根据不同平台调优（GPU 推荐 256；CPU 按核数设置），OKL 源码本身完全一致。

> **OCCA/OKL 参考文档**：OCCA 的完整文档位于 `third_party/occa/docs/`，其中与核函数开发最相关的部分：
> - `third_party/occa/docs/guide/okl/introduction.md` — OKL 基本概念（`@outer`/`@inner`/`@shared`/`@exclusive`）
> - `third_party/occa/docs/guide/okl/loops-in-depth.md` — OKL 循环规则
> - `third_party/occa/docs/guide/okl/attributes.md` — OKL 属性（`@dim`、`@tile` 等）
> - `third_party/occa/docs/api/kernel/` — C++ API（`buildKernel`、`run`、`setRunDims` 等）
> - `third_party/occa/examples/cpp/` — OKL 核函数示例（了解 `@outer`/`@inner`/`@tile`/`@shared` 的实际语法和用法时优先查阅）
>
> **在处理 OCCA/OKL 相关问题时，必须先阅读上述文档确认语法和限制，避免出现幻觉。** 关键规则总结：
> - `@outer` 和 `@inner` 是**裸属性**，不用括号或数字：`for (...; @outer)`，**不是** `@outer(0)`
> - OKL **支持嵌套 `@outer`**（多维 block grid，如 `fd2d.okl`），**也支持顺序 `@outer`**（不嵌套的多个 `@outer` 之间顺序执行）
> - 多个 `@inner` 循环**允许存在**但要求**迭代次数必须相同**
> - `@shared` 数组大小必须为编译时常量（可通过 JIT 宏传入）

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
