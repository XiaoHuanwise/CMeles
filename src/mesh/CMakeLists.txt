target_sources(cmeles_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MeshGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StructuredMeshGenerator.cpp
)
target_include_directories(cmeles_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
