if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.24)
    project(photara_mesh_preprocessor LANGUAGES CXX)
endif()

include(GNUInstallDirs)
find_package(CGAL CONFIG REQUIRED)

if(NOT DEFINED PHOTARA_ENABLE_INSTANT_MESHES)
    option(PHOTARA_ENABLE_INSTANT_MESHES "Fetch Instant Meshes and enable C++ remesh_field_aligned" ON)
endif()

set(PHOTARA_MESH_SOURCES
    src/mesh_ops.cpp
    src/pipeline.cpp
    src/remesh_instant.cpp)

if(PHOTARA_ENABLE_INSTANT_MESHES)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/FetchInstantMeshes.cmake")
endif()

add_library(photara_mesh ${PHOTARA_MESH_SOURCES})
add_library(photara::mesh ALIAS photara_mesh)
target_compile_features(photara_mesh PUBLIC cxx_std_20)
target_include_directories(photara_mesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(photara_mesh
    PUBLIC
        CGAL::CGAL
        photara::render)
if(PHOTARA_ENABLE_INSTANT_MESHES)
    target_link_libraries(photara_mesh PRIVATE InstantMeshesLib)
endif()
target_compile_definitions(photara_mesh
    PUBLIC
        PHOTARA_HAS_INSTANT_MESHES=$<BOOL:${PHOTARA_ENABLE_INSTANT_MESHES}>)

if(MSVC)
    target_compile_options(photara_mesh PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(photara_mesh PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(photara_mesh_preprocessor src/main.cpp)
target_link_libraries(photara_mesh_preprocessor PRIVATE photara::mesh)
install(TARGETS photara_mesh_preprocessor RUNTIME DESTINATION photara_drender/tools)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WIN32)
    find_file(
        PHOTARA_GMP_RUNTIME
        NAMES gmp-10.dll
        HINTS "${GMP_INCLUDE_DIR}/../bin"
        NO_DEFAULT_PATH)
    if(PHOTARA_GMP_RUNTIME)
        install(FILES "${PHOTARA_GMP_RUNTIME}" DESTINATION photara_drender/tools)
    else()
        message(WARNING "gmp-10.dll was not found; the installed CGAL preprocessor may require a runtime DLL")
    endif()
endif()

if(PHOTARA_BUILD_PYTHON_BINDINGS)
    pybind11_add_module(_photara_mesh src/python_bindings.cpp)
    target_link_libraries(_photara_mesh PRIVATE photara::mesh Python::NumPy)
    set_target_properties(_photara_mesh PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/photara_drender"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/photara_drender")
    foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(_photara_mesh PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/python/photara_drender"
            RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/python/photara_drender")
    endforeach()
    install(TARGETS _photara_mesh LIBRARY DESTINATION photara_drender RUNTIME DESTINATION photara_drender)
endif()

if(MSVC)
    target_compile_options(photara_mesh_preprocessor PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(photara_mesh_preprocessor PRIVATE -Wall -Wextra -Wpedantic)
endif()
