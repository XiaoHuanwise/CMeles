# Solver integration tests (isentropic vortex).

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

set(BLAS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/blas/Blas.cpp
)

set(TIME_SOURCES
    ${CMAKE_SOURCE_DIR}/src/time/SimpleExplicitStepper.cpp
    ${CMAKE_SOURCE_DIR}/src/time/RungeKuttaStepper.cpp
    ${CMAKE_SOURCE_DIR}/src/time/ImplicitResidual.cpp
)

set(SOLVER_SOURCES
    ${CMAKE_SOURCE_DIR}/src/dg/DgField.cpp
    ${CMAKE_SOURCE_DIR}/src/riemann/Riemann.cpp
    ${CMAKE_SOURCE_DIR}/src/config/Config.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/CompressibleFlowSolver.cpp
    ${CMAKE_SOURCE_DIR}/src/solver/ExprInitialCondition.cpp
)

add_executable(test_euler_vortex test_euler_vortex.cpp
               ${MESH_SOURCES} ${BASIS_SOURCES} ${CORE_SOURCES}
               ${BLAS_SOURCES} ${TIME_SOURCES} ${SOLVER_SOURCES})
target_include_directories(test_euler_vortex PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_euler_vortex PRIVATE Eigen3::Eigen)
target_link_libraries(test_euler_vortex PRIVATE OCCA::libocca)
target_include_directories(test_euler_vortex PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
target_include_directories(test_euler_vortex PRIVATE ${CMAKE_SOURCE_DIR}/third_party/exprtk)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_euler_vortex PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME euler_vortex COMMAND test_euler_vortex)

add_executable(test_vortex_methods test_vortex_methods.cpp
               ${MESH_SOURCES} ${BASIS_SOURCES} ${CORE_SOURCES}
               ${BLAS_SOURCES} ${TIME_SOURCES} ${SOLVER_SOURCES})
target_include_directories(test_vortex_methods PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_vortex_methods PRIVATE Eigen3::Eigen)
target_link_libraries(test_vortex_methods PRIVATE OCCA::libocca)
target_include_directories(test_vortex_methods PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tomlplusplus)
target_include_directories(test_vortex_methods PRIVATE ${CMAKE_SOURCE_DIR}/third_party/exprtk)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_vortex_methods PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME vortex_methods COMMAND test_vortex_methods)
