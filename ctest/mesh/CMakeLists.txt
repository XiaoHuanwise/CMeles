# Mesh topology/geometry unit tests.

add_executable(test_mesh_geometry test_mesh_geometry.cpp)
target_link_libraries(test_mesh_geometry PRIVATE cmeles_core)
add_test(NAME mesh_geometry COMMAND test_mesh_geometry)

add_executable(test_mesh_device_memory test_mesh_device_memory.cpp)
target_link_libraries(test_mesh_device_memory PRIVATE cmeles_core)
add_test(NAME mesh_device_memory COMMAND test_mesh_device_memory)
