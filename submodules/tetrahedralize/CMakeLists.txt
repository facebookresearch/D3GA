# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16)
project(tetra)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set the CMake module path
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include Libigl
include(libigl)
include_directories("include")

# Enable the target igl::glfw
igl_include(glfw)
igl_include(copyleft tetgen)

# Add source files
file(GLOB SRC_FILES *.cpp src/*.cpp)

# Create the executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link libraries
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl_copyleft::tetgen stdc++fs)
else()
  target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl_copyleft::tetgen)
endif()
