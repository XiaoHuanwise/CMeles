# Discontinuity limiter module.

target_sources(cmeles_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/MrWenoLimiter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PositivityLimiter.cpp
)
target_include_directories(cmeles_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
