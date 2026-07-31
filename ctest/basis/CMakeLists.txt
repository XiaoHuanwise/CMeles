# Basis functions unit tests
# Only built when BUILD_TESTING=ON (default ON).

# The basis sources are compiled into CMeles; we also need them here.
# Compile them directly for standalone test executables.
set(BASIS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/basis/BasisFunctions1D.cpp
    ${CMAKE_SOURCE_DIR}/src/basis/BasisFunctions2D.cpp
)

set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/DeviceMemoryManager.cpp
)
add_executable(test_basis test_basis.cpp ${BASIS_SOURCES})
target_include_directories(test_basis PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_basis PRIVATE Eigen3::Eigen)
target_link_libraries(test_basis PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_basis PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME basis_functions COMMAND test_basis)

# --- test_device_memory: OCCA device memory allocation ---
add_executable(test_device_memory test_device_memory.cpp
               ${BASIS_SOURCES} ${CORE_SOURCES})
target_include_directories(test_device_memory PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_device_memory PRIVATE Eigen3::Eigen)
target_link_libraries(test_device_memory PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_device_memory PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME device_memory COMMAND test_device_memory)
