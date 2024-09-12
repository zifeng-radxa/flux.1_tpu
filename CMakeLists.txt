cmake_minimum_required(VERSION 3.10)
project(sd3 VERSION 3.1.0)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(LIBSOPHON /opt/sophon/libsophon-current)
include_directories(${LIBSOPHON}/include)
link_directories(${LIBSOPHON}/lib)
message(STATUS "USE_LIBSOPHON: ${USE_LIBSOPHON}")

include_directories(src)
# find all cc files in src
file(GLOB CCFile "src/*.cc" "src/models/*.cc")
add_library(sd3 SHARED ${CCFile} )
target_link_libraries(sd3 )
target_link_libraries(sd3 bmlib bmrt libbmodel.a pthread)

# add_executable(test src/sd3.cc )
# target_link_libraries(test bmlib bmrt libbmodel.a pthread)