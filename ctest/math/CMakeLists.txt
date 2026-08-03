set(MATH_SOURCES
    ${CMAKE_SOURCE_DIR}/src/math/MathOps.cpp
)

set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/DeviceMemoryManager.cpp
)

add_executable(test_math test_math.cpp ${MATH_SOURCES} ${CORE_SOURCES})
target_include_directories(test_math PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_math PRIVATE Eigen3::Eigen)
target_link_libraries(test_math PRIVATE OCCA::libocca)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(test_math PRIVATE USE_FLOAT_PRECISION)
endif()
add_test(NAME math COMMAND test_math)
