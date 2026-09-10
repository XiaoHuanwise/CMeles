# Time-marching module tests.

add_executable(test_time_stepper test_time_stepper.cpp)
target_link_libraries(test_time_stepper PRIVATE cmeles_core)
add_test(NAME time_stepper COMMAND test_time_stepper)
