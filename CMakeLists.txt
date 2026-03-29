cmake_minimum_required(VERSION 3.28)
project(BioBrain VERSION 0.1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# --- Dependencies ---

find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Metal framework (macOS)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)

# QCustomPlot (header-only or built from source)
# Users should place qcustomplot.h/cpp in thirdparty/qcustomplot/
set(QCUSTOMPLOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/qcustomplot)

# --- Metal Shader Compilation ---

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

# --- Sources ---

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

set(INPUT_SOURCES
    src/input/WebcamCapture.mm
    src/input/RetinalEncoder.cpp
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
)

set(COMPUTE_SOURCES
    src/compute/CPUBackend.cpp
    src/compute/MetalBackend.mm
)

set(GUI_SOURCES
    src/gui/MainWindow.cpp
    src/gui/RegionTreeWidget.cpp
    src/gui/SpikeRasterWidget.cpp
    src/gui/ActivityMapWidget.cpp
    src/gui/WebcamWidget.cpp
    src/gui/BackendConfigPanel.cpp
    src/gui/NeuronParamEditor.cpp
)

set(RECORDING_SOURCES
    src/recording/SpikeRecorder.cpp
)

# QCustomPlot source (if available)
if(EXISTS ${QCUSTOMPLOT_DIR}/qcustomplot.cpp)
    set(QCUSTOMPLOT_SOURCES ${QCUSTOMPLOT_DIR}/qcustomplot.cpp)
    set(QCUSTOMPLOT_INCLUDE ${QCUSTOMPLOT_DIR})
else()
    message(WARNING "QCustomPlot not found at ${QCUSTOMPLOT_DIR}. Spike raster plotting will be unavailable.")
    set(QCUSTOMPLOT_SOURCES "")
    set(QCUSTOMPLOT_INCLUDE "")
    add_compile_definitions(NO_QCUSTOMPLOT)
endif()

# --- Executable ---

add_executable(BioBrain
    src/main.cpp
    ${CORE_SOURCES}
    ${PLASTICITY_SOURCES}
    ${INPUT_SOURCES}
    ${REGION_SOURCES}
    ${COMPUTE_SOURCES}
    ${GUI_SOURCES}
    ${RECORDING_SOURCES}
    ${QCUSTOMPLOT_SOURCES}
)

add_dependencies(BioBrain MetalShaders)

target_include_directories(BioBrain PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${HDF5_INCLUDE_DIRS}
    ${QCUSTOMPLOT_INCLUDE}
)

target_link_libraries(BioBrain PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    ${HDF5_CXX_LIBRARIES}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${AVFOUNDATION_FRAMEWORK}
    ${COREMEDIA_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
)

# Copy metallib to build directory
add_custom_command(TARGET BioBrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_FILE} $<TARGET_FILE_DIR:BioBrain>/BioBrain.metallib
    COMMENT "Copying Metal shader library to output directory"
)

# --- REST Test Harness (no Qt, no HDF5, no Metal required) ---

set(HARNESS_SOURCES
    src/harness/main_harness.cpp
    src/harness/RestHarness.cpp
)

add_executable(BioBrainHarness
    ${HARNESS_SOURCES}
    ${CORE_SOURCES}
    ${PLASTICITY_SOURCES}
    ${REGION_SOURCES}
    src/compute/CPUBackend.cpp
    src/recording/SpikeRecorder.cpp
)

target_include_directories(BioBrainHarness PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(BioBrainHarness PRIVATE NO_METAL NO_QT)

target_link_libraries(BioBrainHarness PRIVATE
    ${FOUNDATION_FRAMEWORK}
)
