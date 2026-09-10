# Riemann solver module.

target_sources(cmeles_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Riemann.cpp
)
target_include_directories(cmeles_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
