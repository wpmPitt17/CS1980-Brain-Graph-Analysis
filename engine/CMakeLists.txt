cmake_minimum_required(VERSION 3.15)
project(connectome_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly target Python 3.14
set(Python3_ROOT_DIR "/usr")
find_package(Python3 3.14 EXACT COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 via cmake config
find_package(pybind11 REQUIRED CONFIG)

message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
message(STATUS "pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")

# Build the Python extension module
pybind11_add_module(connectome_engine connectome_engine.cpp)

target_include_directories(connectome_engine PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

target_compile_options(connectome_engine PRIVATE -O3)
target_link_libraries(connectome_engine PRIVATE ${Python3_LIBRARIES})