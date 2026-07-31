# CMeles 开发环境配置指南

> **最后编辑日期：2026-07-28**
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

# 若需使用 OpenCL，安装 OpenCL ICD Loader（可选，根据实际需求安装，AMD GPU若使用OpenCL需要安装）
sudo apt install -y ocl-icd-libopencl1 opencl-headers ocl-icd-opencl-dev
```

不要使用 apt 安装 mesa-opencl-icd 来解决 OpenCL 依赖问题，因为 mesa-opencl-icd 使用的是 OpenCL 1.1 的接口，而 cmeles 需要 OpenCL 2.0 以上的接口。

虽然mesa-opencl-icd装了之后可以在occa info看到opencl平台和设备信息，但是CMeles无法通过编译。

应当直接安装显卡的对应驱动。

对于 AMD GPU，参考 [AMDGPU 官方安装指南](https://amdgpu-install.readthedocs.io/en/latest/install-prereq.html#downloading-the-installer-package) 安装驱动。 或者（没尝试过，针对独显或者新核显） [AMD ROCm 官方安装指南](https://rocm.docs.amd.com/en/latest/install/rocm.html) 安装 ROCm 驱动。

若 amdgpu 途径安装完成后，出现 clinfo 无法显示 GPU 信息，但是 sudo clinfo 可以显示 GPU 信息的情况，查阅 [AMDGPU 官方安装指南 OpenCL 相关问题](https://amdgpu-install.readthedocs.io/en/latest/install-installing.html#opencl-optional-component) ，将用户添加到对应用户组，重启后解决。


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
conda install -y -c conda-forge gcc gxx gfortran gcc_impl_linux-64 gxx_impl_linux-64 gfortran_impl_linux-64 eigen hdf5 highfive cmake openblas blas lapack openmp ocl-icd-system
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

### 4.1 libocca

更详细的安装步骤请参考 libocca 的 github 页面：https://github.com/libocca/occa/tree/main

```bash
conda activate cmeles
cd /path/to/CMeles
git submodule update --init --recursive
cd third_party/occa
# 此处示例启用 OpenCL 和 OpenMP 后端，其他后端根据实际需求启用
OCCA_ENABLE_CUDA="OFF" OCCA_ENABLE_HIP="OFF" OCCA_ENABLE_DPCPP="OFF" OCCA_ENABLE_METAL="OFF" OCCA_ENABLE_OPENCL="ON" OCCA_ENABLE_OPENMP="ON" INSTALL_DIR="$(pwd)/../.occa_install" ./configure-cmake.sh
cmake --build build --parallel <number-of-threads>
ctest --test-dir build --output-on-failure
cmake --install build --prefix ../.occa_install
```

按照 libocca 的 installation guide ，需要使用 module load occa 之后才可以 使用 occa info 查看系统硬件信息。需要注意一下几点：
- module load occa 之后可能会导致 ctest 测试中有额外的项目测试失败，解决方法是使用 module unload occa 之后再运行 ctest 测试。
- 若错误安装显卡驱动会导致 occa info 运行时报错，或者不显示显卡硬件信息。
- occa info 采用字符串匹配的形式查看硬件信息，因此需要英语环境才能正确显示硬件信息。可使用命令 LANG=en_US occa info 查看硬件信息。

---

## 5. 环境验证

项目提供了一键验证脚本，用于检查开发环境是否配置正确（cmake、g++、Eigen3、OCCA 及编译运行）。

```bash
conda activate cmeles
cd /path/to/CMeles
bash tests/verify_env.sh
```

验证通过的输出示例：

```
========================================
  CMeles Environment Verification
========================================

[----] Checking basic tools...
[PASS] cmake  (cmake version x.x.x)
[PASS] g++    (g++ (conda-forge gcc x.x.x) x.x.x)

[----] Checking third-party dependencies...
[PASS] OCCA installed
[PASS] Eigen3 installed

[----] CMake configure...
-- Configuring done (x.xs)
-- Generating done (x.xs)
-- Build files have been written to: /path/to/CMeles/tests/build
[PASS] CMake configure succeeded

[----] Building...
[ 50%] Building CXX object CMakeFiles/verify_env.dir/verify_env.cpp.o
[100%] Linking CXX executable verify_env
[100%] Built target verify_env
[PASS] Build succeeded

[----] Running verify_env...
----------------------------------------
===== Eigen Test =====
A =
1 2 3
4 5 6
7 8 9
x = 1 2 3
A * x = 14 32 50

Solve M * x = rhs:
M =
2 1 0
1 3 1
0 1 2
rhs = 1 2 3
x   = 0.5 ...
residual ||M*x - rhs|| = ...

===== OCCA Test =====
OCCA mode: Serial
Vector addition result (a + b):
  0 + 0 = 0 PASS
  1 + 2 = 3 PASS
  ...
OCCA test: PASSED
Eigen test: PASSED
----------------------------------------
[PASS] Run succeeded

========================================
  All checks passed ✓
========================================
```

如果某一步失败，脚本会用 `[FAIL]` 标注具体原因，请根据提示排查。

> **说明**：验证脚本位于 `tests/` 目录下，构建产物在 `tests/build/`，与主工程完全隔离，不影响正式开发。

---

## 6. 常见问题

| 问题                       | 解决方案                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `conda: command not found` | 确认已运行 `conda init bash` 并重启终端                                                                                                                |
| OCCA 编译某后端失败        | OCCA 各后端需要对应的驱动/运行时：CUDA 需安装 CUDA Toolkit；HIP 需安装 ROCm；OpenCL 一般在安装显卡官方驱动时会一并安装。确认对应驱动已正确安装后再编译 |
|                            |                                                                                                                                                        |

---

> 📝 本指南仍在完善中，遇到问题或有补充请直接修改此文档。
