# Riemann solver module tests.

add_executable(test_riemann test_riemann.cpp)
target_link_libraries(test_riemann PRIVATE cmeles_core)
add_test(NAME riemann COMMAND test_riemann)
