# Riemann solver module tests.

add_executable(test_riemann test_riemann.cpp
               ${CMAKE_SOURCE_DIR}/src/riemann/Riemann.cpp
               ${CMAKE_SOURCE_DIR}/src/core/DeviceMemoryManager.cpp)
target_include_directories(test_riemann PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_riemann PRIVATE Eigen3::Eigen)
target_link_libraries(test_riemann PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_riemann PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME riemann COMMAND test_riemann)
