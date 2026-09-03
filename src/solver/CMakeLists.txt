# Top-level solver module (controller + expression initial condition).

target_sources(CMeles PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CompressibleFlowSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ExprInitialCondition.cpp
)
target_include_directories(CMeles PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
