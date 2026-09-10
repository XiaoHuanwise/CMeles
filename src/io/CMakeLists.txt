# I/O module: HDF5 output writers and the offline checkpoint converter.

target_sources(cmeles_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CheckpointConvert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CheckpointWriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldOutput.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HDF5Writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SolutionWriter.cpp
)
target_include_directories(cmeles_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
