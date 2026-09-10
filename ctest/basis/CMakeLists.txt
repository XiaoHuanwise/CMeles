# Basis functions unit tests
# Only built when BUILD_TESTING=ON (default ON).
# cmeles_core carries the module sources, include paths and the
# USE_FLOAT_PRECISION definition.

add_executable(test_basis test_basis.cpp)
target_link_libraries(test_basis PRIVATE cmeles_core)
add_test(NAME basis_functions COMMAND test_basis)

# --- test_device_memory: OCCA device memory allocation ---
add_executable(test_device_memory test_device_memory.cpp)
target_link_libraries(test_device_memory PRIVATE cmeles_core)
add_test(NAME device_memory COMMAND test_device_memory)
