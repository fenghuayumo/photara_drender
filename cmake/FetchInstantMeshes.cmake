# Fetches PyNanoInstantMeshes vendored Instant Meshes core (BSD) and builds InstantMeshesLib.
include(FetchContent)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

set(ASDIFF_INSTANT_MESHES_GIT_TAG d02a725f93a1a66308ae0a63532779ae815ecc32 CACHE STRING
    "Pinned PyNanoInstantMeshes commit providing Instant Meshes sources")

FetchContent_Declare(
    instant_meshes
    URL "https://codeload.github.com/vork/PyNanoInstantMeshes/tar.gz/${ASDIFF_INSTANT_MESHES_GIT_TAG}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# Avoid add_subdirectory on the Python package root; only use the native/ tree.
FetchContent_GetProperties(instant_meshes)
if(NOT instant_meshes_POPULATED)
    FetchContent_Populate(instant_meshes)
endif()

set(ASDIFF_INSTANT_MESHES_ROOT "${instant_meshes_SOURCE_DIR}/native")
set(ASDIFF_INSTANT_MESHES_IM "${ASDIFF_INSTANT_MESHES_ROOT}/im")
set(ASDIFF_INSTANT_MESHES_EXT "${ASDIFF_INSTANT_MESHES_ROOT}/ext")

# Vendored TBB still declares cmake_minimum_required(VERSION 3.1); modern CMake rejects that.
set(_asdiff_tbb_cmake "${ASDIFF_INSTANT_MESHES_EXT}/tbb/CMakeLists.txt")
if(EXISTS "${_asdiff_tbb_cmake}")
    file(READ "${_asdiff_tbb_cmake}" _asdiff_tbb_cmake_text)
    string(REPLACE
        "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)"
        "cmake_minimum_required(VERSION 3.10 FATAL_ERROR)"
        _asdiff_tbb_cmake_text
        "${_asdiff_tbb_cmake_text}")
    file(WRITE "${_asdiff_tbb_cmake}" "${_asdiff_tbb_cmake_text}")
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
        "${ASDIFF_INSTANT_MESHES_EXT}/tbb"
        "${CMAKE_BINARY_DIR}/_deps/instant_meshes_tbb"
        EXCLUDE_FROM_ALL)

    add_library(InstantMeshesLib STATIC
        "${ASDIFF_INSTANT_MESHES_IM}/src/adjacency.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/batch.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/bvh.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/cleanup.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/dedge.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/extract.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/field.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/hierarchy.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/meshio.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/meshstats.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/normal.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/reorder.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/serializer.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/smoothing.cpp"
        "${ASDIFF_INSTANT_MESHES_IM}/src/subdivide.cpp")
    add_library(asdiff::instant_meshes ALIAS InstantMeshesLib)

    target_include_directories(InstantMeshesLib
        PUBLIC
            "${ASDIFF_INSTANT_MESHES_IM}/include"
            "${ASDIFF_INSTANT_MESHES_EXT}/eigen"
            "${ASDIFF_INSTANT_MESHES_EXT}/tbb/include"
            "${ASDIFF_INSTANT_MESHES_EXT}/dset"
            "${ASDIFF_INSTANT_MESHES_EXT}/pss"
            "${ASDIFF_INSTANT_MESHES_EXT}/pcg32"
            "${ASDIFF_INSTANT_MESHES_EXT}/rply"
            "${ASDIFF_INSTANT_MESHES_EXT}/half")
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
