# 0) 先进 cmeles 环境（提供 g++ 和 OpenCL 头/库）【默认开发环境已在cmeles中，不需要激活】
# conda activate cmeles
# 等价写法（脚本里用这个更稳）【对于agent执行来说更合适】：
# export PATH=~/miniconda3/envs/cmeles/bin:$PATH

# 1) 双精度构建（默认 double）
cmake -B build
cmake --build build -j 4

# 2) 单精度构建（float）
cmake -B build_float -DUSE_FLOAT_PRECISION=ON
cmake --build build_float -j 4

# 3) 跑测试
ctest --test-dir build --output-on-failure        # double：14/14，约 80s
ctest --test-dir build_float --output-on-failure  # float：14/14，约 60s

# 4) 主程序用法：CMeles <config_name.toml>
#    配置文件各节的键与默认值见 config/default.toml（输出由 [output] 节控制，默认关闭），例如：
#      ./build/CMeles config/default.toml

# 5) CMelesConvert 用法（离线检查点转换，[output] strategy = "checkpoint" 的配套工具）：
#    批量：转换目录内 mesh.h5 + 全部 checkpoint_<step>.h5 -> solution_<step>.h5
#      ./build/CMelesConvert <directory>
#    单个：转换单个检查点（-o 省略时输出到检查点同目录的 solution_<step>.h5）
#      ./build/CMelesConvert <mesh.h5> <checkpoint_<step>.h5> [-o <out.h5>]
