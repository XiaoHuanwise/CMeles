target_sources(CMeles PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MeshGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StructuredMeshGenerator.cpp
)
target_include_directories(CMeles PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
