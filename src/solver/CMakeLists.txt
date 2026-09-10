# Top-level solver module (controller + expression initial condition).

target_sources(cmeles_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CompressibleFlowSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ExprInitialCondition.cpp
)
target_include_directories(cmeles_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
