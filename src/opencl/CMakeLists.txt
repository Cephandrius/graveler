cmake_minimum_required(VERSION 3.1)

project(graveler LANGUAGES C)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} main.c)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_link_libraries(${PROJECT_NAME} PRIVATE m)
target_link_libraries(${PROJECT_NAME} PRIVATE rt)

set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 11
												 C_STANDARD_REQUIRED ON
												 C_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=200)
