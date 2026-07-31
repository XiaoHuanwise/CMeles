target_sources(CMeles PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/BasisFunctions1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BasisFunctions2D.cpp
)
target_include_directories(CMeles PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
