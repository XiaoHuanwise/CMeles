cmake_minimum_required(VERSION 3.15)
project(CMeles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo, Debug
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/.occa_install")

add_executable(CMeles CMeles.cpp)

# add_subdirectory(src)

find_package(Eigen3 REQUIRED CONFIG)
target_link_libraries(CMeles PRIVATE Eigen3::Eigen)

find_package(OCCA REQUIRED CONFIG)
target_link_libraries(CMeles PRIVATE OCCA::libocca)
