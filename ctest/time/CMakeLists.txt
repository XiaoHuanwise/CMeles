# Time-marching module tests.

set(TIME_SOURCES
    ${CMAKE_SOURCE_DIR}/src/time/SimpleExplicitStepper.cpp
    ${CMAKE_SOURCE_DIR}/src/time/RungeKuttaStepper.cpp
    ${CMAKE_SOURCE_DIR}/src/time/ImplicitResidual.cpp
)

set(BLAS_SOURCES
    ${CMAKE_SOURCE_DIR}/src/blas/Blas.cpp
)

set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/DeviceMemoryManager.cpp
)

add_executable(test_time_stepper test_time_stepper.cpp
               ${TIME_SOURCES} ${BLAS_SOURCES} ${CORE_SOURCES})
target_include_directories(test_time_stepper PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_time_stepper PRIVATE Eigen3::Eigen)
target_link_libraries(test_time_stepper PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_time_stepper PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME time_stepper COMMAND test_time_stepper)
