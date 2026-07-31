# CTest root: optional testing for the CMeles project.
# Enable with: cmake -DBUILD_TESTING=ON .. (default ON)
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(basis)
endif()
