# Config module tests.

add_executable(test_config test_config.cpp
               ${CMAKE_SOURCE_DIR}/src/config/Config.cpp)
target_include_directories(test_config PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_config PRIVATE Eigen3::Eigen)
target_include_directories(test_config PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_config PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME config COMMAND test_config)
