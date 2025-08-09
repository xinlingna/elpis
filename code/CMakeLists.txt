cmake_minimum_required(VERSION 3.5)
project(ELPIS)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx -mavx2 -msse3  -fopenmp ")
include_directories(include)
set(LIBELPIS src/BufferManager.cpp src/calc_utils.cpp src/Hercules.cpp src/Node.cpp src/Index.cpp src/pqueue.cpp src/QueryEngine.cpp src/Setting.cpp src/file_utils.cpp)
add_library(libelpis STATIC ${LIBELPIS})
find_package(Boost REQUIRED COMPONENTS chrono timer system program_options)
add_executable(ELPIS main.cpp )


target_link_libraries(ELPIS libelpis ${Boost_LIBRARIES})



