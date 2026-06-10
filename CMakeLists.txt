cmake_minimum_required(VERSION 3.20)

project(vvm VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

option(VVM_STRICT "Enable strict compiler warnings" ON)
option(VVM_WARN_AS_ERROR "Treat warnings as errors" OFF)
option(VVM_BUILD_CLI "Build CLI entry point" ON)
option(VVM_BUILD_TESTS "Build test programs" ON)
option(VVM_BUILD_VISUALIZER "Build SDL3 visualizer support" OFF)
option(VVM_FETCH_DEPS "Fetch missing dependencies with CMake FetchContent" ON)
set(VVM_SDL_DEPS "auto" CACHE STRING "SDL dependency source: auto, system, or fetch")
set_property(CACHE VVM_SDL_DEPS PROPERTY STRINGS auto system fetch)
if (NOT VVM_SDL_DEPS MATCHES "^(auto|system|fetch)$")
    message(FATAL_ERROR "VVM_SDL_DEPS must be one of: auto, system, fetch")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(vvm_apply_warnings target_name)
    if (NOT VVM_STRICT)
        return()
    endif()

    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
        if (VVM_WARN_AS_ERROR)
            target_compile_options(${target_name} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
        )
        if (VVM_WARN_AS_ERROR)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(vvm_fetch_sdl3)
    set(SDL_X11_XTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        # release-3.4.0
        GIT_TAG a962f40bbba175e9716557a25d5d7965f134a3d3
    )
    FetchContent_MakeAvailable(SDL3)
endfunction()

function(vvm_find_system_sdl3 out_var)
    find_package(SDL3 3.4.0 CONFIG QUIET)
    if (SDL3_FOUND)
        set(${out_var} ON PARENT_SCOPE)
    else()
        set(${out_var} OFF PARENT_SCOPE)
    endif()
endfunction()

if (VVM_BUILD_VISUALIZER)
    set(VVM_EFFECTIVE_SDL_DEPS "${VVM_SDL_DEPS}")
    if (VVM_EFFECTIVE_SDL_DEPS STREQUAL "auto")
        if (VVM_FETCH_DEPS)
            set(VVM_EFFECTIVE_SDL_DEPS "fetch")
        else()
            set(VVM_EFFECTIVE_SDL_DEPS "system")
        endif()
    endif()

    if (VVM_EFFECTIVE_SDL_DEPS STREQUAL "fetch")
        if (NOT VVM_FETCH_DEPS)
            message(FATAL_ERROR "VVM_SDL_DEPS=fetch requires VVM_FETCH_DEPS=ON.")
        endif()
        message(STATUS "Using fetched SDL3 dependency.")
        vvm_fetch_sdl3()
    else()
        vvm_find_system_sdl3(VVM_SYSTEM_SDL3_FOUND)
        if (NOT VVM_SYSTEM_SDL3_FOUND)
            message(FATAL_ERROR "VVM_SDL_DEPS=system requires SDL3. Use VVM_SDL_DEPS=fetch to build the pinned SDL3 dependency.")
        endif()
        message(STATUS "Using system SDL3 dependency.")
    endif()
endif()

add_library(vvm_core STATIC
    src/model.cpp
    src/tasks.cpp
)
add_library(vvm::core ALIAS vvm_core)

target_include_directories(vvm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(vvm_core PUBLIC cxx_std_20)
vvm_apply_warnings(vvm_core)

if (VVM_BUILD_CLI)
    add_executable(vvm
        src/main.cpp
    )
    target_link_libraries(vvm PRIVATE vvm::core)
    target_compile_features(vvm PRIVATE cxx_std_20)
    vvm_apply_warnings(vvm)

    if (VVM_BUILD_VISUALIZER)
        target_sources(vvm PRIVATE
            src/sdl_text.cpp
            src/visualizer_sdl3.cpp
        )
        target_compile_definitions(vvm PRIVATE VVM_WITH_SDL3=1)
        target_link_libraries(vvm PRIVATE SDL3::SDL3)
    endif()
endif()

if (VVM_BUILD_TESTS)
    enable_testing()

    add_executable(vvm_core_tests tests/core_tests.cpp)
    target_link_libraries(vvm_core_tests PRIVATE vvm::core)
    target_compile_features(vvm_core_tests PRIVATE cxx_std_20)
    vvm_apply_warnings(vvm_core_tests)

    add_test(NAME vvm_core_tests COMMAND vvm_core_tests)
endif()
