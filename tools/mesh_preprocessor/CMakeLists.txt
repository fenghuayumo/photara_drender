if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.24)
    project(aether_mesh_preprocessor LANGUAGES CXX)
endif()

include(GNUInstallDirs)
find_package(CGAL CONFIG REQUIRED)

if(NOT DEFINED AETHER_ENABLE_INSTANT_MESHES)
    option(AETHER_ENABLE_INSTANT_MESHES "Fetch Instant Meshes and enable C++ remesh_field_aligned" ON)
endif()

set(AETHER_MESH_SOURCES
    src/mesh_ops.cpp
    src/pipeline.cpp
    src/remesh_instant.cpp)

if(AETHER_ENABLE_INSTANT_MESHES)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/FetchInstantMeshes.cmake")
endif()

add_library(aether_mesh ${AETHER_MESH_SOURCES})
add_library(aether::mesh ALIAS aether_mesh)
target_compile_features(aether_mesh PUBLIC cxx_std_20)
target_include_directories(aether_mesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(aether_mesh
    PUBLIC
        CGAL::CGAL
        aether::render)
if(AETHER_ENABLE_INSTANT_MESHES)
    target_link_libraries(aether_mesh PRIVATE InstantMeshesLib)
endif()
target_compile_definitions(aether_mesh
    PUBLIC
        AETHER_HAS_INSTANT_MESHES=$<BOOL:${AETHER_ENABLE_INSTANT_MESHES}>)

if(MSVC)
    target_compile_options(aether_mesh PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(aether_mesh PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(aether_mesh_preprocessor src/main.cpp)
target_link_libraries(aether_mesh_preprocessor PRIVATE aether::mesh)
install(TARGETS aether_mesh_preprocessor RUNTIME DESTINATION aether_drender/tools)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WIN32)
    find_file(
        AETHER_GMP_RUNTIME
        NAMES gmp-10.dll
        HINTS "${GMP_INCLUDE_DIR}/../bin"
        NO_DEFAULT_PATH)
    if(AETHER_GMP_RUNTIME)
        install(FILES "${AETHER_GMP_RUNTIME}" DESTINATION aether_drender/tools)
    else()
        message(WARNING "gmp-10.dll was not found; the installed CGAL preprocessor may require a runtime DLL")
    endif()
endif()

if(AETHER_BUILD_PYTHON)
    pybind11_add_module(_aether_mesh src/python_bindings.cpp)
    target_link_libraries(_aether_mesh PRIVATE aether::mesh Python::NumPy)
    set_target_properties(_aether_mesh PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/aether_drender"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/aether_drender")
    foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(_aether_mesh PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/python/aether_drender"
            RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/python/aether_drender")
    endforeach()
    install(TARGETS _aether_mesh LIBRARY DESTINATION aether_drender RUNTIME DESTINATION aether_drender)
endif()

if(MSVC)
    target_compile_options(aether_mesh_preprocessor PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(aether_mesh_preprocessor PRIVATE -Wall -Wextra -Wpedantic)
endif()
