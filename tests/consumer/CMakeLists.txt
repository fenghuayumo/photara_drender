cmake_minimum_required(VERSION 3.24)
project(asdiff_render_consumer LANGUAGES CXX)

find_package(asdiff_render 0.1 REQUIRED)
add_executable(asdiff_render_consumer main.cpp)
target_link_libraries(asdiff_render_consumer PRIVATE asdiff::render)

