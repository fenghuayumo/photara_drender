if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.24)
    project(asdiff_mesh_preprocessor LANGUAGES CXX)
endif()

include(GNUInstallDirs)
find_package(CGAL CONFIG REQUIRED)

if(NOT DEFINED ASDIFF_ENABLE_INSTANT_MESHES)
    option(ASDIFF_ENABLE_INSTANT_MESHES "Fetch Instant Meshes and enable C++ remesh_field_aligned" ON)
endif()

set(ASDIFF_MESH_SOURCES
    src/mesh_ops.cpp
    src/pipeline.cpp
    src/remesh_instant.cpp)

if(ASDIFF_ENABLE_INSTANT_MESHES)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/FetchInstantMeshes.cmake")
endif()

add_library(asdiff_mesh ${ASDIFF_MESH_SOURCES})
add_library(asdiff::mesh ALIAS asdiff_mesh)
target_compile_features(asdiff_mesh PUBLIC cxx_std_20)
target_include_directories(asdiff_mesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(asdiff_mesh
    PUBLIC
        CGAL::CGAL
        asdiff::render)
if(ASDIFF_ENABLE_INSTANT_MESHES)
    target_link_libraries(asdiff_mesh PRIVATE InstantMeshesLib)
endif()
target_compile_definitions(asdiff_mesh
    PUBLIC
        ASDIFF_HAS_INSTANT_MESHES=$<BOOL:${ASDIFF_ENABLE_INSTANT_MESHES}>)

if(MSVC)
    target_compile_options(asdiff_mesh PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(asdiff_mesh PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(asdiff_mesh_preprocessor src/main.cpp)
target_link_libraries(asdiff_mesh_preprocessor PRIVATE asdiff::mesh)
install(TARGETS asdiff_mesh_preprocessor RUNTIME DESTINATION asdiff_render/tools)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WIN32)
    find_file(
        ASDIFF_GMP_RUNTIME
        NAMES gmp-10.dll
        HINTS "${GMP_INCLUDE_DIR}/../bin"
        NO_DEFAULT_PATH)
    if(ASDIFF_GMP_RUNTIME)
        install(FILES "${ASDIFF_GMP_RUNTIME}" DESTINATION asdiff_render/tools)
    else()
        message(WARNING "gmp-10.dll was not found; the installed CGAL preprocessor may require a runtime DLL")
    endif()
endif()

if(ASDIFF_BUILD_PYTHON)
    pybind11_add_module(_asdiff_mesh src/python_bindings.cpp)
    target_link_libraries(_asdiff_mesh PRIVATE asdiff::mesh Python::NumPy)
    set_target_properties(_asdiff_mesh PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/asdiff_render"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/asdiff_render")
    foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(_asdiff_mesh PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/python/asdiff_render"
            RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/python/asdiff_render")
    endforeach()
    install(TARGETS _asdiff_mesh LIBRARY DESTINATION asdiff_render RUNTIME DESTINATION asdiff_render)
endif()

if(MSVC)
    target_compile_options(asdiff_mesh_preprocessor PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(asdiff_mesh_preprocessor PRIVATE -Wall -Wextra -Wpedantic)
endif()
