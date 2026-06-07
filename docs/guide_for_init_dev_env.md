# CMeles 开发环境配置指南

> **最后编辑日期：2026-06-07**
>
> ⚠️ **本指南适用于 Linux 环境（Ubuntu 22.04.5 LTS）**，其他发行版或系统可能需要做相应调整。

---

## 目录

1. [系统依赖](#1-系统依赖)
2. [Miniconda3 安装](#2-miniconda3-安装)
3. [Conda 环境配置](#3-conda-环境配置)
4. [项目依赖安装](#4-项目依赖安装)
5. [环境验证](#5-环境验证)
6. [常见问题](#6-常见问题)

---

## 1. 系统依赖

如果已经安装了 Git、curl 和 wget，可以跳过此步骤。

```bash
sudo apt update
sudo apt install -y git curl wget
```

---

## 2. Miniconda3 安装

### 2.1 下载安装脚本

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 2.2 执行安装

```bash
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：
- 按 `Enter` 阅读许可协议，输入 `yes` 确认
- 确认安装路径（默认 `~/miniconda3`，一般直接回车即可）
- 安装结束时询问是否初始化，输入 `yes`

安装完成后，添加 `conda-forge` 频道（社区维护的频道，包含大量官方频道没有的包）：

```bash
conda config --add channels conda-forge
```

### 2.3 使 conda 生效

```bash
source ~/.bashrc
```

此时终端提示符前应出现 `(base)` 字样，表示安装成功。

### 2.4 配置国内镜像源（可选，推荐）

```bash
# TODO: 根据实际使用的镜像源填写
# 例如清华大学 TUNA 镜像：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
```

### 2.5 验证安装

```bash
conda --version
```

---

## 3. Conda 环境配置

### 3.1 创建项目虚拟环境

```bash
conda create -n cmeles python=3.14 -y
```

### 3.2 激活环境

```bash
conda activate cmeles
```

### 3.3 配置conda环境
```bash
# 如果项目没有提供 environment.yml，可以手动安装必要的依赖：
conda install -y -c conda-forge gcc gxx gfortran gcc_impl_linux-64 gxx_impl_linux-64 gfortran_impl_linux-64 cmake openblas blas lapack openmp ocl-icd
```


```bash
# 如果项目已提供 environment.yml：
conda env update -f environment.yml --prune
```

添加环境变量（需要重新进入环境）
```bash
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CC=$CONDA_PREFIX/bin/gcc CXX=$CONDA_PREFIX/bin/g++ FC=$CONDA_PREFIX/bin/gfortran -n cmeles
```

重新进入环境
```bash
conda deactivate
conda activate cmeles
```

---

## 4. 项目依赖安装

### 4.1 pip 依赖

```bash
# TODO: 确认实际的 requirements 文件名
pip install -r requirements.txt
# 或
pip install -r requirements-dev.txt
```

### 4.2 项目本身（可编辑模式）

```bash
pip install -e .
```

### 4.3 其他工具链

```bash
# TODO: 列出需要额外安装的工具
# 例如: pre-commit, linter, formatter 等
# pre-commit install
```

---

## 5. 环境验证

```bash
# TODO: 编写验证脚本或命令，确认环境配置正确
# 例如：
# python -c "import cmeles; print(cmeles.__version__)"
# pytest tests/ --co   # 仅收集测试，不执行
```

---

## 6. 常见问题

| 问题                       | 解决方案                                |
| -------------------------- | --------------------------------------- |
| `conda: command not found` | 确认已运行 `conda init bash` 并重启终端 |
| TODO: 补充其他常见问题     | TODO: 对应解决方案                      |
|                            |                                         |

---

> 📝 本指南仍在完善中，遇到问题或有补充请直接修改此文档。
