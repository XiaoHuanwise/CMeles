# DG field module tests.

add_executable(test_dg_field test_dg_field.cpp)
target_link_libraries(test_dg_field PRIVATE cmeles_core)
add_test(NAME dg_field COMMAND test_dg_field)

add_executable(test_dg_periodic test_dg_periodic.cpp)
target_link_libraries(test_dg_periodic PRIVATE cmeles_core)
add_test(NAME dg_periodic COMMAND test_dg_periodic)
