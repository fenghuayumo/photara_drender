cmake_minimum_required(VERSION 3.24)

project(asdiff_render VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ASDIFF_BUILD_PYTHON "Build the Python extension" ON)
option(ASDIFF_BUILD_TESTS "Build native tests" ON)
option(ASDIFF_ENABLE_VALIDATION "Enable Vulkan validation by default" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan 1.2 REQUIRED)
find_program(ASDIFF_DXC_EXECUTABLE
    NAMES dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED)

set(ASDIFF_SHADER_SOURCES
    shaders/rasterize_forward.hlsl
    shaders/rasterize_backward.hlsl
    shaders/reduce_vertex_gradients.hlsl
    shaders/interpolate_forward.hlsl
    shaders/interpolate_grad_attributes.hlsl
    shaders/interpolate_grad_raster.hlsl)

set(ASDIFF_SHADER_BINARIES)
set(ASDIFF_SHADER_EMBEDDED_HEADERS)
foreach(shader IN LISTS ASDIFF_SHADER_SOURCES)
    get_filename_component(shader_name "${shader}" NAME)
    set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv")
    add_custom_command(
        OUTPUT "${shader_binary}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${ASDIFF_DXC_EXECUTABLE}"
                -spirv -T cs_6_0 -E main -O3
                -fspv-target-env=vulkan1.2
                -fvk-use-dx-layout
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
    src/rasterizer.cpp)
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
target_compile_definitions(asdiff_render
    PRIVATE
        ASDIFF_ENABLE_VALIDATION=$<BOOL:${ASDIFF_ENABLE_VALIDATION}>)

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
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_FILE_DIR:_asdiff_render>/shaders")
    foreach(shader_binary IN LISTS ASDIFF_SHADER_BINARIES)
        add_custom_command(TARGET _asdiff_render POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${shader_binary}"
                    "$<TARGET_FILE_DIR:_asdiff_render>/shaders/")
    endforeach()
    install(TARGETS _asdiff_render LIBRARY DESTINATION asdiff_render RUNTIME DESTINATION asdiff_render)
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

install(TARGETS asdiff_render EXPORT asdiff_render_targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${ASDIFF_SHADER_BINARIES} DESTINATION ${CMAKE_INSTALL_DATADIR}/asdiff_render/shaders)
install(
    EXPORT asdiff_render_targets
    FILE asdiff_renderTargets.cmake
    NAMESPACE asdiff::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/asdiff_render)
configure_package_config_file(
    cmake/asdiff_renderConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/asdiff_render)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/asdiff_renderConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/asdiff_render)
