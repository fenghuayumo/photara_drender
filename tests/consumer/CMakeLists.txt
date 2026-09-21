cmake_minimum_required(VERSION 3.24)
project(photara_drender_consumer LANGUAGES CXX)

find_package(photara_drender 0.1 REQUIRED)
add_executable(photara_drender_consumer main.cpp)
target_link_libraries(photara_drender_consumer PRIVATE photara::render)

