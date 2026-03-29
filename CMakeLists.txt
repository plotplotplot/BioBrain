cmake_minimum_required(VERSION 3.28)

# Enable CUDA if available (must be before project())
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(BioBrain VERSION 0.1.0 LANGUAGES CXX CUDA)
    set(HAS_CUDA TRUE)
    message(STATUS "CUDA compiler found — NVIDIA GPU acceleration enabled")
elseif(APPLE)
    project(BioBrain VERSION 0.1.0 LANGUAGES CXX OBJCXX)
    set(HAS_CUDA FALSE)
else()
    project(BioBrain VERSION 0.1.0 LANGUAGES CXX)
    set(HAS_CUDA FALSE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Address Sanitizer for debug builds
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer ENABLED")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ═══════════════════════════════════════════════════════════════════════════════
# Platform Detection
# ═══════════════════════════════════════════════════════════════════════════════

if(APPLE)
    set(PLATFORM_MACOS TRUE)
    message(STATUS "Platform: macOS (Metal + AVFoundation + CoreAudio)")
elseif(UNIX AND NOT APPLE)
    set(PLATFORM_LINUX TRUE)
    message(STATUS "Platform: Linux (${HAS_CUDA} CUDA + V4L2 + PulseAudio)")
else()
    message(FATAL_ERROR "Unsupported platform. BioBrain requires macOS or Linux.")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Dependencies (cross-platform)
# ═══════════════════════════════════════════════════════════════════════════════

find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia)
find_package(HDF5 COMPONENTS CXX)

# QCustomPlot
set(QCUSTOMPLOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/qcustomplot)
if(EXISTS ${QCUSTOMPLOT_DIR}/qcustomplot.cpp)
    set(QCUSTOMPLOT_SOURCES ${QCUSTOMPLOT_DIR}/qcustomplot.cpp)
    set(QCUSTOMPLOT_INCLUDE ${QCUSTOMPLOT_DIR})
else()
    message(WARNING "QCustomPlot not found at ${QCUSTOMPLOT_DIR}.")
    set(QCUSTOMPLOT_SOURCES "")
    set(QCUSTOMPLOT_INCLUDE "")
    add_compile_definitions(NO_QCUSTOMPLOT)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# macOS-specific: Metal, AVFoundation, CoreAudio
# ═══════════════════════════════════════════════════════════════════════════════

if(PLATFORM_MACOS)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
    find_library(COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
    find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox REQUIRED)
    find_library(AUDIOUNIT_FRAMEWORK AudioUnit REQUIRED)

    # Metal shader compilation (optional — requires Xcode)
    execute_process(
        COMMAND xcrun -sdk macosx --find metal
        RESULT_VARIABLE METAL_COMPILER_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(METAL_COMPILER_RESULT EQUAL 0)
        set(HAS_METAL_COMPILER TRUE)
        message(STATUS "Metal shader compiler found")

        set(METAL_SHADERS
            src/metal/izhikevich.metal
            src/metal/hodgkin_huxley.metal
            src/metal/adex.metal
            src/metal/synapse_update.metal
        )
        set(METAL_AIR_FILES)
        foreach(SHADER ${METAL_SHADERS})
            get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
            set(AIR_FILE ${CMAKE_BINARY_DIR}/${SHADER_NAME}.air)
            add_custom_command(
                OUTPUT ${AIR_FILE}
                COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/${SHADER} -o ${AIR_FILE}
                DEPENDS ${CMAKE_SOURCE_DIR}/${SHADER}
                COMMENT "Compiling Metal shader: ${SHADER_NAME}"
            )
            list(APPEND METAL_AIR_FILES ${AIR_FILE})
        endforeach()

        set(METALLIB_FILE ${CMAKE_BINARY_DIR}/BioBrain.metallib)
        add_custom_command(
            OUTPUT ${METALLIB_FILE}
            COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METALLIB_FILE}
            DEPENDS ${METAL_AIR_FILES}
            COMMENT "Linking Metal shader library"
        )
        add_custom_target(MetalShaders DEPENDS ${METALLIB_FILE})
    else()
        set(HAS_METAL_COMPILER FALSE)
        message(WARNING "Metal shader compiler not found. Building CPU-only on macOS.")
        add_compile_definitions(NO_METAL)
        add_custom_target(MetalShaders)
    endif()

    set(PLATFORM_INPUT_SOURCES src/input/WebcamCapture.mm)
    set(PLATFORM_AUDIO_SOURCES src/audio/VocalSynthesizer.mm)
    set(PLATFORM_COMPUTE_SOURCES src/compute/MetalBackend.mm)
    set(PLATFORM_LIBS
        ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
        ${AVFOUNDATION_FRAMEWORK} ${COREMEDIA_FRAMEWORK} ${COREVIDEO_FRAMEWORK}
        ${AUDIOTOOLBOX_FRAMEWORK} ${AUDIOUNIT_FRAMEWORK}
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Linux-specific: V4L2, PulseAudio, CUDA
# ═══════════════════════════════════════════════════════════════════════════════

if(PLATFORM_LINUX)
    add_compile_definitions(NO_METAL PLATFORM_LINUX)
    add_custom_target(MetalShaders)  # no-op

    # V4L2 (usually in libc on Linux, no extra library)
    set(PLATFORM_INPUT_SOURCES src/input/WebcamCapture_linux.cpp)

    # PulseAudio
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(PULSE libpulse-simple)
    endif()
    if(PULSE_FOUND)
        message(STATUS "PulseAudio found")
        set(PLATFORM_AUDIO_SOURCES src/audio/VocalSynthesizer_linux.cpp)
    else()
        message(WARNING "PulseAudio not found. Audio output disabled.")
        set(PLATFORM_AUDIO_SOURCES "")
        add_compile_definitions(NO_AUDIO)
    endif()

    # CUDA backend
    if(HAS_CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Detect GPU architecture (RTX 3060 = sm_86)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90" CACHE STRING "CUDA architectures")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        set(CUDA_KERNEL_SOURCES
            src/cuda/izhikevich.cu
            src/cuda/hodgkin_huxley.cu
            src/cuda/adex.cu
            src/cuda/synapse_update.cu
        )
        set(PLATFORM_COMPUTE_SOURCES
            src/compute/CUDABackend.cu
        )
    else()
        message(WARNING "CUDA not found. Building CPU-only on Linux.")
        set(CUDA_KERNEL_SOURCES "")
        set(PLATFORM_COMPUTE_SOURCES "")
        add_compile_definitions(NO_CUDA)
    endif()

    set(PLATFORM_LIBS "")
    if(PULSE_FOUND)
        list(APPEND PLATFORM_LIBS ${PULSE_LIBRARIES})
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Core Sources (cross-platform)
# ═══════════════════════════════════════════════════════════════════════════════

set(CORE_SOURCES
    src/core/IzhikevichNeuron.cpp
    src/core/HodgkinHuxleyNeuron.cpp
    src/core/AdExNeuron.cpp
    src/core/LIFNeuron.cpp
    src/core/Synapse.cpp
    src/core/SpikeRouter.cpp
    src/core/BrainRegion.cpp
    src/core/Simulation.cpp
)

set(PLASTICITY_SOURCES
    src/plasticity/STDP.cpp
    src/plasticity/DopamineSTDP.cpp
    src/plasticity/NeuromodulatorySTDP.cpp
)

set(REGION_SOURCES
    src/regions/Retina.cpp
    src/regions/LGN.cpp
    src/regions/V1.cpp
    src/regions/V2V4.cpp
    src/regions/ITCortex.cpp
    src/regions/VTA.cpp
    src/regions/Striatum.cpp
    src/regions/MotorCortex.cpp
    src/regions/WernickesArea.cpp
    src/regions/BrocasArea.cpp
)

set(GUI_SOURCES
    src/gui/MainWindow.cpp
    src/gui/RegionTreeWidget.cpp
    src/gui/SpikeRasterWidget.cpp
    src/gui/ActivityMapWidget.cpp
    src/gui/WebcamWidget.cpp
    src/gui/WebcamPanel.cpp
    src/gui/BackendConfigPanel.cpp
    src/gui/NeuronParamEditor.cpp
)

set(RECORDING_SOURCES
    src/recording/SpikeRecorder.cpp
)

set(HARNESS_API_SOURCES
    src/harness/RestHarness.cpp
    src/harness/DebugAPI.cpp
)

# ═══════════════════════════════════════════════════════════════════════════════
# Main Executable
# ═══════════════════════════════════════════════════════════════════════════════

add_executable(BioBrain
    src/main.cpp
    ${CORE_SOURCES}
    ${PLASTICITY_SOURCES}
    ${PLATFORM_INPUT_SOURCES}
    src/input/RetinalEncoder.cpp
    ${REGION_SOURCES}
    src/compute/CPUBackend.cpp
    ${PLATFORM_COMPUTE_SOURCES}
    ${CUDA_KERNEL_SOURCES}
    ${GUI_SOURCES}
    ${RECORDING_SOURCES}
    ${PLATFORM_AUDIO_SOURCES}
    ${HARNESS_API_SOURCES}
    ${QCUSTOMPLOT_SOURCES}
)

add_dependencies(BioBrain MetalShaders)

target_include_directories(BioBrain PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${QCUSTOMPLOT_INCLUDE}
)
if(HDF5_FOUND)
    target_include_directories(BioBrain PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(BioBrain PRIVATE ${HDF5_CXX_LIBRARIES})
    target_compile_definitions(BioBrain PRIVATE HAS_HDF5)
endif()
if(PLATFORM_LINUX AND PULSE_FOUND)
    target_include_directories(BioBrain PRIVATE ${PULSE_INCLUDE_DIRS})
endif()

target_link_libraries(BioBrain PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    ${PLATFORM_LIBS}
)

# macOS app bundle
if(PLATFORM_MACOS)
    set_target_properties(BioBrain PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/Info.plist
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.biobrain.app"
        MACOSX_BUNDLE_BUNDLE_NAME "BioBrain"
    )
    if(HAS_METAL_COMPILER)
        add_custom_command(TARGET BioBrain POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_FILE} $<TARGET_FILE_DIR:BioBrain>/BioBrain.metallib
            COMMENT "Copying Metal shader library to output directory"
        )
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# REST Test Harness (minimal dependencies — works on both platforms)
# ═══════════════════════════════════════════════════════════════════════════════

add_executable(BioBrainHarness
    src/harness/main_harness.cpp
    src/harness/RestHarness.cpp
    ${CORE_SOURCES}
    ${PLASTICITY_SOURCES}
    ${REGION_SOURCES}
    src/compute/CPUBackend.cpp
    src/recording/SpikeRecorder.cpp
)

target_include_directories(BioBrainHarness PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(BioBrainHarness PRIVATE NO_METAL NO_CUDA NO_QT)

if(PLATFORM_MACOS)
    target_link_libraries(BioBrainHarness PRIVATE ${FOUNDATION_FRAMEWORK})
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "BioBrain Build Configuration:")
message(STATUS "  Platform:     ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  GPU Backend:  ${HAS_CUDA} CUDA | ${PLATFORM_MACOS} Metal")
message(STATUS "  Webcam:       ${PLATFORM_MACOS} AVFoundation | ${PLATFORM_LINUX} V4L2")
message(STATUS "  Audio:        ${PLATFORM_MACOS} CoreAudio | ${PLATFORM_LINUX} PulseAudio")
message(STATUS "  HDF5:         ${HDF5_FOUND}")
message(STATUS "  QCustomPlot:  ${QCUSTOMPLOT_SOURCES}")
message(STATUS "")
