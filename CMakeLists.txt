cmake_minimum_required(VERSION 2.8)
project(Sick3Core C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package( OpenCV REQUIRED )
find_package( grt REQUIRED )

include_directories(${GRT_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GRT_LIBRARIES})

add_executable(Sick3Core Sick3Core.cpp)
target_link_libraries(Sick3Core ${OpenCV_LIBS} ${LIBS})

