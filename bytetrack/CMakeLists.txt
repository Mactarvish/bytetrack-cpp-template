project(bytetrack)

file(GLOB src_cpps ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
					)

add_library(${PROJECT_NAME} SHARED 
            ${src_cpps}
)

add_compile_definitions(BYTETRACK_LIBRARY_EXPORTS)

set_target_properties (${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX d)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIR}
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>" "${CMAKE_CURRENT_BINARY_DIR}/../bytetrack_demo/"
    COMMAND ${CMAKE_COMMAND} -E echo "COPY $<TARGET_FILE_DIR:${PROJECT_NAME}> TO ${CMAKE_CURRENT_BINARY_DIR}/../bytetrack_demo/"
)

