cmake_minimum_required(VERSION 3.24)
project(MiniDLSS LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options ---
option(MINI_DLSS_ENABLE_CUDA "Build with CUDA acceleration" OFF)
option(MINI_DLSS_ENABLE_GUI  "Build with Dear ImGui GUI"     ON)

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# --- CUDA (optional) ---
if(MINI_DLSS_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
endif()

# --- GUI dependencies (optional) ---
if(MINI_DLSS_ENABLE_GUI)
    include(FetchContent)

    # GLFW
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.9)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)

    # Dear ImGui
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.90.4)
    FetchContent_MakeAvailable(imgui)

    # Build ImGui as a static library
    add_library(imgui_lib STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends)
    target_link_libraries(imgui_lib PUBLIC glfw)
    if(WIN32)
        target_link_libraries(imgui_lib PUBLIC opengl32)
    else()
        find_package(OpenGL REQUIRED)
        target_link_libraries(imgui_lib PUBLIC OpenGL::GL)
    endif()

    # tinyfiledialogs (vendored — official source is SourceForge, not GitHub)
    set(TINYFD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyfiledialogs)
    add_library(tinyfiledialogs_lib STATIC
        ${TINYFD_DIR}/tinyfiledialogs.c)
    target_include_directories(tinyfiledialogs_lib PUBLIC
        ${TINYFD_DIR})
endif()

# --- Source files ---
set(CORE_SOURCES
    src/flow.cpp
    src/warp.cpp
    src/blend.cpp
    src/video_io.cpp
    src/pipeline.cpp
    src/cuda_backend.cpp)

set(APP_SOURCES
    src/main.cpp
    src/app.cpp)

if(MINI_DLSS_ENABLE_GUI)
    list(APPEND APP_SOURCES src/gui.cpp)
endif()

# Platform-specific sources
if(WIN32)
    list(APPEND APP_SOURCES src/systray.cpp)
endif()

# --- CUDA sources ---
if(MINI_DLSS_ENABLE_CUDA)
    set(CUDA_SOURCES
        cuda/warp_host.cu
        cuda/blend_host.cu
        cuda/occlusion_kernel.cu)
endif()

# --- Executable ---
if(WIN32 AND MINI_DLSS_ENABLE_GUI)
    add_executable(MiniDLSS WIN32 ${CORE_SOURCES} ${APP_SOURCES} ${CUDA_SOURCES})
else()
    add_executable(MiniDLSS ${CORE_SOURCES} ${APP_SOURCES} ${CUDA_SOURCES})
endif()

target_include_directories(MiniDLSS PRIVATE src/)

# --- Preprocessor defines ---
if(MINI_DLSS_ENABLE_CUDA)
    target_compile_definitions(MiniDLSS PRIVATE HAS_CUDA=1)
    target_include_directories(MiniDLSS PRIVATE cuda/)
    target_link_libraries(MiniDLSS PRIVATE CUDA::cudart)
else()
    target_compile_definitions(MiniDLSS PRIVATE HAS_CUDA=0)
endif()

if(MINI_DLSS_ENABLE_GUI)
    target_compile_definitions(MiniDLSS PRIVATE HAS_GUI=1)
    target_link_libraries(MiniDLSS PRIVATE imgui_lib tinyfiledialogs_lib)
else()
    target_compile_definitions(MiniDLSS PRIVATE HAS_GUI=0)
endif()

# --- Link libraries ---
target_link_libraries(MiniDLSS PRIVATE ${OpenCV_LIBS})

if(WIN32)
    target_link_libraries(MiniDLSS PRIVATE Shell32 User32)
endif()

# --- Install ---
install(TARGETS MiniDLSS RUNTIME DESTINATION .)

# --- CPack ---
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "MiniDLSS")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Mini-DLSS Frame Interpolation")
include(CPack)
