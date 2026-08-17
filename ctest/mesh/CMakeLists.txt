# Mesh topology/geometry unit tests.

set(MESH_SOURCES
    ${CMAKE_SOURCE_DIR}/src/mesh/Mesh.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshGeometry.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/StructuredMeshGenerator.cpp
)

set(BASIS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/basis/BasisFunctions1D.cpp
    ${CMAKE_SOURCE_DIR}/src/basis/BasisFunctions2D.cpp
)

set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/DeviceMemoryManager.cpp
)

add_executable(test_mesh_geometry test_mesh_geometry.cpp
               ${MESH_SOURCES} ${BASIS_SOURCES} ${CORE_SOURCES})
target_include_directories(test_mesh_geometry PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_mesh_geometry PRIVATE Eigen3::Eigen)
target_link_libraries(test_mesh_geometry PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_mesh_geometry PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME mesh_geometry COMMAND test_mesh_geometry)

add_executable(test_mesh_device_memory test_mesh_device_memory.cpp
               ${MESH_SOURCES} ${BASIS_SOURCES} ${CORE_SOURCES})
target_include_directories(test_mesh_device_memory PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_mesh_device_memory PRIVATE Eigen3::Eigen)
target_link_libraries(test_mesh_device_memory PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_mesh_device_memory PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME mesh_device_memory COMMAND test_mesh_device_memory)
