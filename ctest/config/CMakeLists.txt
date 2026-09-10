# Config module tests.

add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE cmeles_core)
add_test(NAME config COMMAND test_config)
