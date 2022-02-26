cmake_minimum_required(VERSION 3.16)
project(anglersim)

set(HEADER_FILES fishpop.h)
set(SOURCE_FILES main.cpp fishpop.cpp ${HEADER_FILES})

add_executable(anglersim ${SOURCE_FILES})

add_definitions(-DHAVE_CBLAS=1)

if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

target_link_libraries(anglersim ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})