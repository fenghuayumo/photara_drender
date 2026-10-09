function(photara_bundle_windows_runtime)
    if(MSVC)
        # Use the compiler's redistributable binaries, never Debug_NonRedist.
        set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
        set(CMAKE_INSTALL_DEBUG_LIBRARIES FALSE)
        set(CMAKE_INSTALL_DEBUG_LIBRARIES_ONLY FALSE)
        include(InstallRequiredSystemLibraries)
        if(NOT CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
            message(FATAL_ERROR "MSVC redistributable DLLs were not found")
        endif()
        foreach(runtime IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
            if(NOT EXISTS "${runtime}")
                message(FATAL_ERROR "Required redistributable DLL is missing: ${runtime}")
            endif()
        endforeach()
        install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION photara_drender)
    endif()

    # Allow a supplied redistributable loader; otherwise use the installed
    # loader matching the Python architecture. Driver ICDs stay on the host.
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(system_directory "$ENV{SystemRoot}/System32")
        set(sdk_directory "$ENV{VULKAN_SDK}/Bin")
    else()
        set(system_directory "$ENV{SystemRoot}/SysWOW64")
        set(sdk_directory "$ENV{VULKAN_SDK}/Bin32")
    endif()
    find_file(PHOTARA_VULKAN_RUNTIME_DLL NAMES vulkan-1.dll
        HINTS "${sdk_directory}" "${system_directory}"
        NO_DEFAULT_PATH REQUIRED)
    install(FILES "${PHOTARA_VULKAN_RUNTIME_DLL}" DESTINATION photara_drender)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/licenses/Vulkan-Loader-LICENSE.txt"
        DESTINATION photara_drender/licenses)
endfunction()
