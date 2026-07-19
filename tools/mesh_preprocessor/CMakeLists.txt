if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.24)
    project(asdiff_mesh_preprocessor LANGUAGES CXX)
endif()

find_package(CGAL CONFIG REQUIRED)

add_executable(asdiff_mesh_preprocessor main.cpp)
target_compile_features(asdiff_mesh_preprocessor PRIVATE cxx_std_20)
target_link_libraries(asdiff_mesh_preprocessor PRIVATE CGAL::CGAL)
install(TARGETS asdiff_mesh_preprocessor RUNTIME DESTINATION asdiff_render/tools)
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

if(MSVC)
    target_compile_options(asdiff_mesh_preprocessor PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(asdiff_mesh_preprocessor PRIVATE -Wall -Wextra -Wpedantic)
endif()
