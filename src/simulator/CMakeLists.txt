cmake_minimum_required(VERSION 3.16)
project(anglersim)

set(HEADER_FILES fishpop.h dynamicalpop.h)
set(SOURCE_FILES main.cpp fishpop.cpp dynamicalpop.cpp ${HEADER_FILES})

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

message(STATUS "###  MPI active ###")
find_package(MPI REQUIRED)
message(STATUS "MPI FOUND: ${MPI_FOUND}")
message(STATUS "include path: ${MPI_INCLUDE_PATH}")
include_directories(${MPI_INCLUDE_PATH})
message(STATUS "mpi compile flags: ${MPI_CXX_COMPILE_FLAGS}")
message(STATUS "cxx flags (before): ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
message(STATUS "cxx flags (after): ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")

target_link_libraries(anglersim ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_C_LIBRARIES})


