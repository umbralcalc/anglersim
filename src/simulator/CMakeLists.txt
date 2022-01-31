cmake_minimum_required(VERSION 3.16)
project(anglersim)

set(HEADER_FILES fishpop.h)
set(SOURCE_FILES main.cpp fishpop.cpp ${HEADER_FILES})

add_executable(anglersim ${SOURCE_FILES})