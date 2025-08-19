cmake_minimum_required(VERSION 3.12)
project(dedup_cpp_core CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
# Check for AVX2 support and add the flag if available
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "Compiler supports -mavx2. Enabling AVX2 optimizations.")
    add_compile_options(-march=native)
else()
    message(WARNING "Compiler does not support -mavx2. AVX2 optimizations will be disabled.")
endif()

add_subdirectory(third_party/abseil-cpp)

set(CMAKE_BUILD_TYPE Release)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED HINTS "${pybind11_CMAKE_DIR}")

# your path for faiss
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

find_path(FAISS_INCLUDE_DIR
          NAMES "faiss/Index.h"
          HINTS ${CMAKE_PREFIX_PATH}
          # PATH_SUFFIXES
          PATH_SUFFIXES include
          # NO_DEFAULT_PATH
          NO_DEFAULT_PATH
)


find_library(FAISS_LIBRARY
             NAMES faiss
             HINTS ${CMAKE_PREFIX_PATH}
             PATH_SUFFIXES lib lib64
             NO_DEFAULT_PATH
)


if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIBRARY)
    message(FATAL_ERROR "Could not find Faiss headers or library. "
                        "Please provide the Faiss installation path via CMAKE_PREFIX_PATH. "
                        "Current CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
endif()

message(STATUS "SUCCESS: Found Faiss include directory: ${FAISS_INCLUDE_DIR}")
message(STATUS "SUCCESS: Found Faiss library: ${FAISS_LIBRARY}")
# ===================================================================


set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

get_filename_component(PYTHON_LIB_DIR ${Python3_LIBRARIES} DIRECTORY)

set(CMAKE_INSTALL_RPATH "${PYTHON_LIB_DIR}")

set(SOURCES dedup_cpp/dedup.cpp)


add_library(_core MODULE ${SOURCES})



set_target_properties(_core PROPERTIES PREFIX "")

target_include_directories(_core PRIVATE
    ${FAISS_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/dedup_cpp # Add xxhash.h path
)

target_link_libraries(_core PRIVATE
    ${FAISS_LIBRARY}
    OpenMP::OpenMP_CXX
    pybind11::module
    absl::flat_hash_map
    absl::flat_hash_set
)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_core PRIVATE -fvisibility=default)
endif()


install(TARGETS _core
        LIBRARY DESTINATION .
)