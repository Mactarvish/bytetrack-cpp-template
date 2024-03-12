project(bytetrack_demo VERSION 1.0 DESCRIPTION "")

find_package(OpenCV REQUIRED)

find_package(jsoncpp CONFIG REQUIRED)
set(json_components jsoncpp_lib jsoncpp_object JsonCpp::JsonCpp)

set(d "$<IF:$<CONFIG:Debug>,d,>")

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
								)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

set(json_components jsoncpp_lib jsoncpp_object JsonCpp::JsonCpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${json_components})
target_include_directories(${PROJECT_NAME} PUBLIC 
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                    ${bytetrack_include_dir}
													${OpenCV_INCLUDE_DIRS}
													${EIGEN3_INCLUDE_DIR}
												)

target_link_libraries(${PROJECT_NAME} PRIVATE
						${CMAKE_CURRENT_BINARY_DIR}/${prefix}bytetrack${d}.${postfix})

