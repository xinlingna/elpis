# cmake_minimum_required(VERSION 3.5)
# project(ELPIS)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS_DEBUG "-g")

# # added -fopenmp
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx -mavx2 -msse3  -fopenmp ")
# include_directories(include)
# set(LIBELPIS src/BufferManager.cpp src/calc_utils.cpp src/Hercules.cpp src/Node.cpp src/Index.cpp src/pqueue.cpp src/QueryEngine.cpp src/Setting.cpp src/file_utils.cpp)
# add_library(libelpis STATIC ${LIBELPIS})
# find_package(Boost REQUIRED COMPONENTS chrono timer system program_options)
# add_executable(ELPIS main.cpp )


# target_link_libraries(ELPIS libelpis ${Boost_LIBRARIES})



cmake_minimum_required(VERSION 3.5)
project(ELPIS)

# C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 调试符号
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# 编译选项（保持你的原有设置）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx -mavx2 -msse3 -fopenmp")

# 头文件目录
include_directories(include)

# 源文件
set(LIBELPIS
    src/BufferManager.cpp
    src/calc_utils.cpp
    src/Hercules.cpp
    src/Node.cpp
    src/Index.cpp
    src/pqueue.cpp
    src/QueryEngine.cpp
    src/Setting.cpp
    src/file_utils.cpp
)

# 依赖：Boost（保持你原来的用法）
find_package(Boost REQUIRED COMPONENTS chrono timer system program_options)

# 依赖：oneTBB（关键新增）
find_package(TBB REQUIRED)

# 如果需要，也可以显式查找线程库（多数情况下 TBB 会自己带上 pthread，这里加上更稳）
find_package(Threads REQUIRED)

# 静态库
add_library(libelpis STATIC ${LIBELPIS})

# 关键：在产生 TBB 依赖的库上 **PUBLIC** 链接 TBB（和 pthread）
target_link_libraries(libelpis
    PUBLIC
        TBB::tbb
        Threads::Threads
)

# 可执行文件
add_executable(ELPIS main.cpp)

# 链接：可执行文件链接你的静态库与 Boost
# 因为 libelpis 对 TBB 是 PUBLIC，ELPIS 会自动继承 -ltbb
target_link_libraries(ELPIS
    PRIVATE
        libelpis
        ${Boost_LIBRARIES}
)
