cmake_minimum_required(VERSION 3.20)
project(Test LANGUAGES CXX)

# Generates compile_commands.json for clangd / VS Code C++ (F12 Go to Definition).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# Layout: sources under develop/Test, 3rdParty and sibling builds under develop/
set(DEVELOP_ROOT "/home/roots/develop" CACHE PATH "Parent of 3rdParty and build/")
set(THIRD_PARTY_DIR "${DEVELOP_ROOT}/3rdParty" CACHE PATH "Third-party source tree")
set(BUILD_ROOT "${DEVELOP_ROOT}/build" CACHE PATH "CMake install prefix for built deps (glfw, OpenMesh, …)")

list(APPEND CMAKE_PREFIX_PATH
    "${BUILD_ROOT}/glfw"
    "${BUILD_ROOT}/OpenMesh"
)

set(APP_HEADERS
    "${CMAKE_SOURCE_DIR}/VulkanWindow.h"
    "${CMAKE_SOURCE_DIR}/MainWindow.h"
    "${CMAKE_SOURCE_DIR}/makehuman.h"
)

set(TEST_SOURCES
    "${CMAKE_SOURCE_DIR}/Test.h"
    "${CMAKE_SOURCE_DIR}/Test.cpp"
    "${CMAKE_SOURCE_DIR}/TestGPU.cu"
)

set(SHADER_FILES
    "${CMAKE_SOURCE_DIR}/shaders/shader.vert"
    "${CMAKE_SOURCE_DIR}/shaders/shader.frag"
    "${CMAKE_SOURCE_DIR}/shaders/ruler.vert"
    "${CMAKE_SOURCE_DIR}/shaders/ruler.frag"
)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME}
    main.cpp
    VulkanWindow.cpp
    MainWindow.cpp
    makehuman.cpp
    ${TEST_SOURCES}
    ${APP_HEADERS}
    ${SHADER_FILES}
)

source_group("Include Files" FILES ${APP_HEADERS})
source_group("Test Files" FILES ${TEST_SOURCES})
source_group("Shader Files" FILES ${SHADER_FILES})
if(MSVC)
    set_source_files_properties("${CMAKE_SOURCE_DIR}/TestGPU.cu" PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "/TP"
    )
else()
    set_source_files_properties("${CMAKE_SOURCE_DIR}/TestGPU.cu" PROPERTIES LANGUAGE CXX)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)

if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE /ignore:4099 /ignore:4819)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    "${THIRD_PARTY_DIR}/glm"
    "${THIRD_PARTY_DIR}/eigen"
    "${THIRD_PARTY_DIR}/OpenMesh/src"
    "${THIRD_PARTY_DIR}/freetype/include"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Vulkan::Vulkan
)

# Refresh packaged SPIR-V when glslc is available so GPU shaders match the GLSL in shaders/.
find_program(VULKAN_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")
if(VULKAN_GLSLC)
    set(_MH_VERT_SPV "${CMAKE_SOURCE_DIR}/shaders/shader.vert.spv")
    set(_MH_FRAG_SPV "${CMAKE_SOURCE_DIR}/shaders/shader.frag.spv")
    add_custom_command(
        OUTPUT "${_MH_VERT_SPV}" "${_MH_FRAG_SPV}"
        COMMAND "${VULKAN_GLSLC}" "${CMAKE_SOURCE_DIR}/shaders/shader.vert" -o "${_MH_VERT_SPV}"
        COMMAND "${VULKAN_GLSLC}" "${CMAKE_SOURCE_DIR}/shaders/shader.frag" -o "${_MH_FRAG_SPV}"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/shader.vert" "${CMAKE_SOURCE_DIR}/shaders/shader.frag"
        COMMENT "Compiling shaders/shader.{vert,frag} to SPIR-V")
    add_custom_target(update_main_shaders DEPENDS "${_MH_VERT_SPV}" "${_MH_FRAG_SPV}")
    add_dependencies(${PROJECT_NAME} update_main_shaders)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE TEST_USE_PREBUILT_SPV_ONLY)

find_package(OpenMesh QUIET CONFIG)
if(TARGET OpenMeshCore::OpenMeshCore)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMeshCore::OpenMeshCore)
elseif(TARGET OpenMesh::OpenMeshCore)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMesh::OpenMeshCore)
else()
    find_library(OPENMESH_CORE_LIB NAMES OpenMeshCore libOpenMeshCore
        PATHS "${BUILD_ROOT}/OpenMesh" "${BUILD_ROOT}/OpenMesh/Build" "${BUILD_ROOT}/OpenMesh/lib"
        PATH_SUFFIXES lib lib/Release lib/Debug
    )
    if(OPENMESH_CORE_LIB)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENMESH_CORE_LIB})
    endif()
endif()

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Comdlg32)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
