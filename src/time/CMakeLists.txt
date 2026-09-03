# Time-marching module.

target_sources(CMeles PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleExplicitStepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RungeKuttaStepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DualStepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitResidual.cpp
)
target_include_directories(CMeles PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
