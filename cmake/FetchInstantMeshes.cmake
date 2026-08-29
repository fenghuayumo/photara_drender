# Fetches PyNanoInstantMeshes vendored Instant Meshes core (BSD) and builds InstantMeshesLib.
include(FetchContent)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

set(AETHER_INSTANT_MESHES_GIT_TAG d02a725f93a1a66308ae0a63532779ae815ecc32 CACHE STRING
    "Pinned PyNanoInstantMeshes commit providing Instant Meshes sources")

FetchContent_Declare(
    instant_meshes
    URL "https://codeload.github.com/vork/PyNanoInstantMeshes/tar.gz/${AETHER_INSTANT_MESHES_GIT_TAG}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# Avoid add_subdirectory on the Python package root; only use the native/ tree.
FetchContent_GetProperties(instant_meshes)
if(NOT instant_meshes_POPULATED)
    FetchContent_Populate(instant_meshes)
endif()

set(AETHER_INSTANT_MESHES_ROOT "${instant_meshes_SOURCE_DIR}/native")
set(AETHER_INSTANT_MESHES_IM "${AETHER_INSTANT_MESHES_ROOT}/im")
set(AETHER_INSTANT_MESHES_EXT "${AETHER_INSTANT_MESHES_ROOT}/ext")

# Vendored TBB still declares cmake_minimum_required(VERSION 3.1); modern CMake rejects that.
set(_aether_tbb_cmake "${AETHER_INSTANT_MESHES_EXT}/tbb/CMakeLists.txt")
if(EXISTS "${_aether_tbb_cmake}")
    file(READ "${_aether_tbb_cmake}" _aether_tbb_cmake_text)
    string(REPLACE
        "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)"
        "cmake_minimum_required(VERSION 3.10 FATAL_ERROR)"
        _aether_tbb_cmake_text
        "${_aether_tbb_cmake_text}")
    file(WRITE "${_aether_tbb_cmake}" "${_aether_tbb_cmake_text}")
endif()

if(NOT TARGET InstantMeshesLib)
    if(MSVC)
        add_compile_definitions(_CRT_SECURE_NO_WARNINGS __TBB_NO_IMPLICIT_LINKAGE)
    endif()

    set(TBB_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(TBB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
    set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
    set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(
        "${AETHER_INSTANT_MESHES_EXT}/tbb"
        "${CMAKE_BINARY_DIR}/_deps/instant_meshes_tbb"
        EXCLUDE_FROM_ALL)

    add_library(InstantMeshesLib STATIC
        "${AETHER_INSTANT_MESHES_IM}/src/adjacency.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/batch.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/bvh.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/cleanup.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/dedge.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/extract.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/field.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/hierarchy.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/meshio.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/meshstats.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/normal.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/reorder.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/serializer.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/smoothing.cpp"
        "${AETHER_INSTANT_MESHES_IM}/src/subdivide.cpp")
    add_library(aether::instant_meshes ALIAS InstantMeshesLib)

    target_include_directories(InstantMeshesLib
        PUBLIC
            "${AETHER_INSTANT_MESHES_IM}/include"
            "${AETHER_INSTANT_MESHES_EXT}/eigen"
            "${AETHER_INSTANT_MESHES_EXT}/tbb/include"
            "${AETHER_INSTANT_MESHES_EXT}/dset"
            "${AETHER_INSTANT_MESHES_EXT}/pss"
            "${AETHER_INSTANT_MESHES_EXT}/pcg32"
            "${AETHER_INSTANT_MESHES_EXT}/rply"
            "${AETHER_INSTANT_MESHES_EXT}/half")
    target_link_libraries(InstantMeshesLib PUBLIC tbb_static)
    target_compile_features(InstantMeshesLib PRIVATE cxx_std_14)
    set_target_properties(InstantMeshesLib PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON)
    # MSVC removed std::unary_function in newer CRT even under /std:c++14.
    target_compile_definitions(InstantMeshesLib PRIVATE _HAS_AUTO_PTR_ETC=1)
    if(MSVC)
        target_compile_options(InstantMeshesLib PRIVATE /W0 /MP /std:c++14)
    else()
        target_compile_options(InstantMeshesLib PRIVATE -w)
    endif()
endif()
