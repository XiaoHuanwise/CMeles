cmake_minimum_required(VERSION 3.15)
project(CMelesVerifyEnv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/.occa_install")

add_executable(verify_env verify_env.cpp)

find_package(Eigen3 REQUIRED CONFIG)
target_link_libraries(verify_env PRIVATE Eigen3::Eigen)

find_package(OCCA REQUIRED CONFIG)
target_link_libraries(verify_env PRIVATE OCCA::libocca)
