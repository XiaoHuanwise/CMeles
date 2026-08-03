# BLAS unit tests
# Only built when BUILD_TESTING=ON.

set(BLAS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/blas/Blas.cpp
)

set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/DeviceMemoryManager.cpp
)

add_executable(test_blas test_blas.cpp ${BLAS_SOURCES} ${CORE_SOURCES})
target_include_directories(test_blas PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_blas PRIVATE Eigen3::Eigen)
target_link_libraries(test_blas PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_blas PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME blas COMMAND test_blas)
