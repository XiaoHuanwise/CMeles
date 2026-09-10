# BLAS unit tests
# Only built when BUILD_TESTING=ON.

add_executable(test_blas test_blas.cpp)
target_link_libraries(test_blas PRIVATE cmeles_core)
add_test(NAME blas COMMAND test_blas)
