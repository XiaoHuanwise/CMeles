add_executable(test_hdf5_output test_hdf5_output.cpp)
target_link_libraries(test_hdf5_output PRIVATE cmeles_core)
add_test(NAME io COMMAND test_hdf5_output)
