# Solver integration tests (isentropic vortex).
# Both executables share VortexCommon.hpp from this directory.

add_executable(test_euler_vortex test_euler_vortex.cpp)
target_link_libraries(test_euler_vortex PRIVATE cmeles_core)
add_test(NAME euler_vortex COMMAND test_euler_vortex)

add_executable(test_vortex_methods test_vortex_methods.cpp)
target_link_libraries(test_vortex_methods PRIVATE cmeles_core)
add_test(NAME vortex_methods COMMAND test_vortex_methods)
