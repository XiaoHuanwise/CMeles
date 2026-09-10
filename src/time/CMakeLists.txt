# Time-marching module.

target_sources(cmeles_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleExplicitStepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RungeKuttaStepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DualStepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitResidual.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StepperFactory.cpp
)
target_include_directories(cmeles_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
