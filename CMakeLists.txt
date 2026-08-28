cmake_minimum_required(VERSION 3.16)
project(Limitless_cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

enable_testing()

add_library(lmlc_tensor STATIC
    src/LMLC/LMLC_Tensor/LMLC_Tensor.cpp
)
target_include_directories(lmlc_tensor PUBLIC src/LMLC/LMLC_Tensor)

add_subdirectory(src/LMLC/LMLC_Tensor/test)
