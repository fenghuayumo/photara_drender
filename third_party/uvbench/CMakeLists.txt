cmake_minimum_required(VERSION 3.21)
project(uvbench LANGUAGES CXX)

set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(UVATLAS_USE_OPENMP OFF CACHE BOOL "" FORCE)
set(UVATLAS_ENABLE_TIMING ON CACHE BOOL "Enable UVAtlas phase timing traces" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../UVAtlas uvatlas)

add_executable(uvbench uvbench.cpp)
target_link_libraries(uvbench PRIVATE UVAtlas)
target_compile_features(uvbench PRIVATE cxx_std_17)
target_compile_definitions(uvbench PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
