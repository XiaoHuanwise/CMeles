# Limiter invariant tests.

add_executable(test_mr_weno test_mr_weno.cpp)
target_link_libraries(test_mr_weno PRIVATE cmeles_core)
add_test(NAME mr_weno COMMAND test_mr_weno)

add_executable(test_positivity test_positivity.cpp)
target_link_libraries(test_positivity PRIVATE cmeles_core)
add_test(NAME positivity COMMAND test_positivity)
