cmake_minimum_required(VERSION 3.16)
project(bytetrack_cpp_template)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(bytetrack_include_dir ${CMAKE_SOURCE_DIR}/bytetrack/include)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(postfix "lib")
    set(prefix "")
else()
    set(postfix "so")
    set(prefix "lib")
endif()

set(d "$<IF:$<CONFIG:Debug>,d,>")
add_compile_definitions("$<IF:$<CONFIG:Debug>,DEBUG,>")

set(CMAKE_MODULE_PATH  "D:/third_party/eigen/cmake")
find_package(Eigen3 REQUIRED)


add_subdirectory(bytetrack)
add_subdirectory(bytetrack_demo)
add_dependencies(bytetrack_demo bytetrack)

install(TARGETS bytetrack)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/bytetrack/include DESTINATION ${CMAKE_INSTALL_PREFIX})

