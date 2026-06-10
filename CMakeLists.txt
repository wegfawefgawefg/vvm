cmake_minimum_required(VERSION 3.20)

project(canned-nnvm VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

option(NNVM_STRICT "Enable strict compiler warnings" ON)
option(NNVM_WARN_AS_ERROR "Treat warnings as errors" OFF)
option(NNVM_BUILD_CLI "Build CLI entry point" ON)
option(NNVM_BUILD_TESTS "Build test programs" ON)
option(NNVM_BUILD_VISUALIZER "Build SDL3 visualizer support" OFF)
option(NNVM_FETCH_DEPS "Fetch missing dependencies with CMake FetchContent" ON)
set(NNVM_SDL_DEPS "auto" CACHE STRING "SDL dependency source: auto, system, or fetch")
set_property(CACHE NNVM_SDL_DEPS PROPERTY STRINGS auto system fetch)
if (NOT NNVM_SDL_DEPS MATCHES "^(auto|system|fetch)$")
    message(FATAL_ERROR "NNVM_SDL_DEPS must be one of: auto, system, fetch")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(nnvm_apply_warnings target_name)
    if (NOT NNVM_STRICT)
        return()
    endif()

    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
        if (NNVM_WARN_AS_ERROR)
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
        if (NNVM_WARN_AS_ERROR)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(nnvm_fetch_sdl3)
    set(SDL_X11_XTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        # release-3.4.0
        GIT_TAG a962f40bbba175e9716557a25d5d7965f134a3d3
    )
    FetchContent_MakeAvailable(SDL3)
endfunction()

function(nnvm_find_system_sdl3 out_var)
    find_package(SDL3 3.4.0 CONFIG QUIET)
    if (SDL3_FOUND)
        set(${out_var} ON PARENT_SCOPE)
    else()
        set(${out_var} OFF PARENT_SCOPE)
    endif()
endfunction()

if (NNVM_BUILD_VISUALIZER)
    set(NNVM_EFFECTIVE_SDL_DEPS "${NNVM_SDL_DEPS}")
    if (NNVM_EFFECTIVE_SDL_DEPS STREQUAL "auto")
        if (NNVM_FETCH_DEPS)
            set(NNVM_EFFECTIVE_SDL_DEPS "fetch")
        else()
            set(NNVM_EFFECTIVE_SDL_DEPS "system")
        endif()
    endif()

    if (NNVM_EFFECTIVE_SDL_DEPS STREQUAL "fetch")
        if (NOT NNVM_FETCH_DEPS)
            message(FATAL_ERROR "NNVM_SDL_DEPS=fetch requires NNVM_FETCH_DEPS=ON.")
        endif()
        message(STATUS "Using fetched SDL3 dependency.")
        nnvm_fetch_sdl3()
    else()
        nnvm_find_system_sdl3(NNVM_SYSTEM_SDL3_FOUND)
        if (NOT NNVM_SYSTEM_SDL3_FOUND)
            message(FATAL_ERROR "NNVM_SDL_DEPS=system requires SDL3. Use NNVM_SDL_DEPS=fetch to build the pinned SDL3 dependency.")
        endif()
        message(STATUS "Using system SDL3 dependency.")
    endif()
endif()

add_library(nnvm_core STATIC
    src/model.cpp
)
add_library(nnvm::core ALIAS nnvm_core)

target_include_directories(nnvm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(nnvm_core PUBLIC cxx_std_20)
nnvm_apply_warnings(nnvm_core)

if (NNVM_BUILD_CLI)
    add_executable(nnvm
        src/main.cpp
    )
    target_link_libraries(nnvm PRIVATE nnvm::core)
    target_compile_features(nnvm PRIVATE cxx_std_20)
    nnvm_apply_warnings(nnvm)

    if (NNVM_BUILD_VISUALIZER)
        target_sources(nnvm PRIVATE src/visualizer_sdl3.cpp)
        target_compile_definitions(nnvm PRIVATE NNVM_WITH_SDL3=1)
        target_link_libraries(nnvm PRIVATE SDL3::SDL3)
    endif()
endif()

if (NNVM_BUILD_TESTS)
    enable_testing()

    add_executable(nnvm_core_tests tests/core_tests.cpp)
    target_link_libraries(nnvm_core_tests PRIVATE nnvm::core)
    target_compile_features(nnvm_core_tests PRIVATE cxx_std_20)
    nnvm_apply_warnings(nnvm_core_tests)

    add_test(NAME nnvm_core_tests COMMAND nnvm_core_tests)
endif()
