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
ctest --test-dir build --output-on-failure        # double：13/13，约 86s
ctest --test-dir build_float --output-on-failure  # float：13/13，约 65s
