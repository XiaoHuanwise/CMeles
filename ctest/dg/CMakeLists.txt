# DG field module tests.

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

set(DG_SOURCES
    ${CMAKE_SOURCE_DIR}/src/dg/DgField.cpp
    ${CMAKE_SOURCE_DIR}/src/riemann/Riemann.cpp
    ${CMAKE_SOURCE_DIR}/src/config/Config.cpp
)

add_executable(test_dg_field test_dg_field.cpp
               ${MESH_SOURCES} ${BASIS_SOURCES} ${CORE_SOURCES} ${DG_SOURCES})
target_include_directories(test_dg_field PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dg_field PRIVATE Eigen3::Eigen)
target_link_libraries(test_dg_field PRIVATE OCCA::libocca)
target_include_directories(test_dg_field PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_dg_field PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME dg_field COMMAND test_dg_field)

add_executable(test_dg_periodic test_dg_periodic.cpp
               ${MESH_SOURCES} ${BASIS_SOURCES} ${CORE_SOURCES} ${DG_SOURCES})
target_include_directories(test_dg_periodic PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dg_periodic PRIVATE Eigen3::Eigen)
target_link_libraries(test_dg_periodic PRIVATE OCCA::libocca)
target_include_directories(test_dg_periodic PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_dg_periodic PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME dg_periodic COMMAND test_dg_periodic)
