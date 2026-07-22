cmake_minimum_required(VERSION 3.24)

project(asdiff_render VERSION 0.4.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ASDIFF_BUILD_PYTHON "Build the Python extension" ON)
option(ASDIFF_BUILD_TESTS "Build native tests" ON)
option(ASDIFF_ENABLE_VALIDATION "Enable Vulkan validation by default" OFF)
option(ASDIFF_ENABLE_UVATLAS "Enable Microsoft UVAtlas unwrapping" ON)
option(ASDIFF_FETCH_UVATLAS "Fetch a pinned UVAtlas source release when it is not installed" ON)
option(ASDIFF_UVATLAS_USE_OPENMP "Enable UVAtlas internal OpenMP (disable when using parallel partitions)" OFF)
option(ASDIFF_BUILD_MESH_TOOLS "Build GPL/commercial CGAL mesh preprocessing tools" OFF)
option(ASDIFF_ENABLE_INSTANT_MESHES "Fetch Instant Meshes and enable C++ remesh_field_aligned" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan 1.2 REQUIRED)
find_program(ASDIFF_DXC_EXECUTABLE
    NAMES dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED)

set(ASDIFF_HAS_UVATLAS OFF)
if(ASDIFF_ENABLE_UVATLAS)
    find_package(uvatlas CONFIG QUIET)
    if(TARGET Microsoft::UVAtlas)
        set(ASDIFF_UVATLAS_TARGET Microsoft::UVAtlas)
        set(ASDIFF_HAS_UVATLAS ON)
    elseif(WIN32 AND ASDIFF_FETCH_UVATLAS)
        include(FetchContent)
        set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(UVATLAS_USE_OPENMP ${ASDIFF_UVATLAS_USE_OPENMP} CACHE BOOL "" FORCE)
        FetchContent_Declare(
            uvatlas
            URL https://codeload.github.com/microsoft/UVAtlas/tar.gz/refs/tags/oct2025
            URL_HASH SHA256=13c2d0d1cbff2924a2aa1e24748ef10c161a6e54ac8eb0e5a3f42b8d54e471af
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(uvatlas)
        set(ASDIFF_UVATLAS_TARGET UVAtlas)
        set(ASDIFF_HAS_UVATLAS ON)
    else()
        message(WARNING "Microsoft UVAtlas was not found; unwrap_uv() will report that the backend is unavailable")
    endif()
endif()

set(ASDIFF_SHADER_SOURCES
    shaders/rasterize_forward.hlsl
    shaders/rasterize_backward.hlsl
    shaders/reduce_vertex_gradients.hlsl
    shaders/interpolate_forward.hlsl
    shaders/interpolate_grad_attributes.hlsl
    shaders/interpolate_grad_raster.hlsl
    shaders/texture_forward.hlsl
    shaders/texture_grad_texture.hlsl
    shaders/texture_grad_uv.hlsl)
list(APPEND ASDIFF_SHADER_SOURCES
    shaders/project_texture.hlsl
    shaders/project_ray_visibility.hlsl
    shaders/atlas_photometric_gradient.hlsl
    shaders/atlas_seam_gradient.hlsl
    shaders/atlas_adam_update.hlsl)

set(ASDIFF_SHADER_BINARIES)
set(ASDIFF_SHADER_EMBEDDED_HEADERS)
foreach(shader IN LISTS ASDIFF_SHADER_SOURCES)
    get_filename_component(shader_name "${shader}" NAME)
    set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv")
    set(shader_profile cs_6_0)
    set(shader_extra_options)
    if(shader_name STREQUAL "project_ray_visibility.hlsl")
        set(shader_profile cs_6_5)
        list(APPEND shader_extra_options -fspv-extension=SPV_KHR_ray_query)
    elseif(shader_name STREQUAL "atlas_photometric_gradient.hlsl")
        set(shader_profile cs_6_6)
    endif()
    add_custom_command(
        OUTPUT "${shader_binary}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${ASDIFF_DXC_EXECUTABLE}"
                -spirv -T ${shader_profile} -E main -O3
                -fspv-target-env=vulkan1.2
                -fvk-use-dx-layout
                ${shader_extra_options}
                "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
                -Fo "${shader_binary}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
        VERBATIM)
    list(APPEND ASDIFF_SHADER_BINARIES "${shader_binary}")
    string(MAKE_C_IDENTIFIER "${shader_name}_spv" shader_variable)
    set(shader_embedded_header "${CMAKE_CURRENT_BINARY_DIR}/generated/${shader_name}.embedded.hpp")
    add_custom_command(
        OUTPUT "${shader_embedded_header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT_FILE=${shader_binary}
                -DOUTPUT_FILE=${shader_embedded_header}
                -DVARIABLE_NAME=${shader_variable}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spir_v.cmake"
        DEPENDS "${shader_binary}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spir_v.cmake"
        VERBATIM)
    list(APPEND ASDIFF_SHADER_EMBEDDED_HEADERS "${shader_embedded_header}")
endforeach()

add_custom_target(asdiff_shaders DEPENDS ${ASDIFF_SHADER_BINARIES} ${ASDIFF_SHADER_EMBEDDED_HEADERS})

add_library(asdiff_render
    src/context.cpp
    src/rasterizer.cpp
    src/texture_baker.cpp
    src/texture_refiner.cpp
    src/uv_atlas.cpp)
add_library(asdiff::render ALIAS asdiff_render)
add_dependencies(asdiff_render asdiff_shaders)
set_target_properties(asdiff_render PROPERTIES EXPORT_NAME render)
target_compile_features(asdiff_render PUBLIC cxx_std_20)

target_include_directories(asdiff_render
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(asdiff_render PUBLIC Vulkan::Vulkan)
if(ASDIFF_HAS_UVATLAS)
    target_link_libraries(asdiff_render PRIVATE ${ASDIFF_UVATLAS_TARGET})
endif()
target_compile_definitions(asdiff_render
    PRIVATE
        ASDIFF_ENABLE_VALIDATION=$<BOOL:${ASDIFF_ENABLE_VALIDATION}>
        ASDIFF_HAS_UVATLAS=$<BOOL:${ASDIFF_HAS_UVATLAS}>)

if(MSVC)
    target_compile_options(asdiff_render PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(asdiff_render PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ASDIFF_BUILD_PYTHON)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_result)
    if(NOT pybind11_result EQUAL 0)
        message(FATAL_ERROR "pybind11 is required when ASDIFF_BUILD_PYTHON=ON")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")
    find_package(pybind11 CONFIG REQUIRED)

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render/__init__.py"
        COPYONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/torch.py"
        "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render/torch.py"
        COPYONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/atlas.py"
        "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render/atlas.py"
        COPYONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/baking.py"
        "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render/baking.py"
        COPYONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/colmap.py"
        "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render/colmap.py"
        COPYONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/mesh_processing.py"
        "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render/mesh_processing.py"
        COPYONLY)

    pybind11_add_module(_asdiff_render python/bindings.cpp)
    target_link_libraries(_asdiff_render PRIVATE asdiff::render Python::NumPy)
    set_target_properties(_asdiff_render PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render")
    foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(_asdiff_render PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${configuration} "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render"
            RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_CURRENT_BINARY_DIR}/python/asdiff_render")
    endforeach()

    add_custom_command(TARGET _asdiff_render POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/__init__.py"
                "$<TARGET_FILE_DIR:_asdiff_render>/__init__.py"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/torch.py"
                "$<TARGET_FILE_DIR:_asdiff_render>/torch.py"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/atlas.py"
                "$<TARGET_FILE_DIR:_asdiff_render>/atlas.py"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/baking.py"
                "$<TARGET_FILE_DIR:_asdiff_render>/baking.py"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/colmap.py"
                "$<TARGET_FILE_DIR:_asdiff_render>/colmap.py"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/python/asdiff_render/mesh_processing.py"
                "$<TARGET_FILE_DIR:_asdiff_render>/mesh_processing.py"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_FILE_DIR:_asdiff_render>/shaders")
    foreach(shader_binary IN LISTS ASDIFF_SHADER_BINARIES)
        add_custom_command(TARGET _asdiff_render POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${shader_binary}"
                    "$<TARGET_FILE_DIR:_asdiff_render>/shaders/")
    endforeach()
    install(TARGETS _asdiff_render LIBRARY DESTINATION asdiff_render RUNTIME DESTINATION asdiff_render)
    install(
        FILES
            python/asdiff_render/__init__.py
            python/asdiff_render/torch.py
            python/asdiff_render/atlas.py
            python/asdiff_render/baking.py
            python/asdiff_render/colmap.py
            python/asdiff_render/mesh_processing.py
        DESTINATION asdiff_render)
    install(FILES ${ASDIFF_SHADER_BINARIES} DESTINATION asdiff_render/shaders)
endif()

if(ASDIFF_BUILD_TESTS)
    enable_testing()
    add_executable(asdiff_native_tests tests/native_tests.cpp)
    target_link_libraries(asdiff_native_tests PRIVATE asdiff::render)
    add_test(NAME asdiff_native_tests COMMAND asdiff_native_tests)
    if(ASDIFF_BUILD_PYTHON)
        add_test(
            NAME asdiff_python_smoke
            COMMAND "${CMAKE_COMMAND}" -E env
                    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
                    "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_smoke.py")
    endif()
endif()

if(ASDIFF_BUILD_MESH_TOOLS)
    add_subdirectory(tools/mesh_preprocessor)
endif()

install(TARGETS asdiff_render EXPORT asdiff_render_targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    FILES LICENSE THIRD_PARTY_NOTICES.md third_party/UVAtlas.LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/asdiff_render)
install(FILES ${ASDIFF_SHADER_BINARIES} DESTINATION ${CMAKE_INSTALL_DATADIR}/asdiff_render/shaders)
if(ASDIFF_HAS_UVATLAS)
    set(ASDIFF_UVATLAS_CONFIG_DEPENDENCY "find_dependency(uvatlas CONFIG)")
else()
    set(ASDIFF_UVATLAS_CONFIG_DEPENDENCY "")
endif()
configure_package_config_file(
    cmake/asdiff_renderConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/asdiff_render)
install(
    EXPORT asdiff_render_targets
    FILE asdiff_renderTargets.cmake
    NAMESPACE asdiff::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/asdiff_render)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/asdiff_render)
