cmake_minimum_required(VERSION 3.24)
project(aether_drender_consumer LANGUAGES CXX)

find_package(aether_drender 0.1 REQUIRED)
add_executable(aether_drender_consumer main.cpp)
target_link_libraries(aether_drender_consumer PRIVATE aether::render)

