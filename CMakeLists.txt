cmake_minimum_required(VERSION 3.17)
project(test_onnxruntime)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-1.14.0/include)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/onnxruntime-1.14.0/libonnxruntime.1.14.0.dylib)

add_subdirectory(yolov5-lite)
add_subdirectory(yolo-nas)

add_executable(test_onnxruntime main.cpp)

target_link_libraries(test_onnxruntime onnxruntime)
target_link_libraries(test_onnxruntime ${OpenCV_LIBS})
target_link_libraries(test_onnxruntime ${yolo_nas_lib})
target_link_libraries(test_onnxruntime ${yolov5_lite_lib})


