cmake_minimum_required(VERSION 3.17)
project(test_onnxruntime)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-1.17.0/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-tts/sherpa-onnx/include)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/onnxruntime-1.17.0/libonnxruntime.1.17.0.dylib)

#add_library(libcargs SHARED IMPORTED)
#set_target_properties(libcargs PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libcargs.dylib)
#add_library(libespeak-ng SHARED IMPORTED)
#set_target_properties(libespeak-ng PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libespeak-ng.dylib)
add_library(libkaldi-decoder-core SHARED IMPORTED)
set_target_properties(libkaldi-decoder-core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libkaldi-decoder-core.dylib)
add_library(libkaldi-native-fbank-core SHARED IMPORTED)
set_target_properties(libkaldi-native-fbank-core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libkaldi-native-fbank-core.dylib)
add_library(libpiper_phonemize.1.2.0 SHARED IMPORTED)
set_target_properties(libpiper_phonemize.1.2.0 PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libpiper_phonemize.1.2.0.dylib)
#add_library(libpiper_phonemize.1 SHARED IMPORTED)
#set_target_properties(libpiper_phonemize.1 PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libpiper_phonemize.1.dylib)
#add_library(libpiper_phonemize SHARED IMPORTED)
#set_target_properties(libpiper_phonemize PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libpiper_phonemize.dylib)
add_library(libsherpa-onnx-c-api SHARED IMPORTED)
set_target_properties(libsherpa-onnx-c-api PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libsherpa-onnx-c-api.dylib)
add_library(libsherpa-onnx-core SHARED IMPORTED)
set_target_properties(libsherpa-onnx-core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libsherpa-onnx-core.dylib)
add_library(libsherpa-onnx-fst.6 SHARED IMPORTED)
set_target_properties(libsherpa-onnx-fst.6 PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libsherpa-onnx-fst.6.dylib)
#add_library(libsherpa-onnx-fst SHARED IMPORTED)
#set_target_properties(libsherpa-onnx-fst PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libsherpa-onnx-fst.dylib)
add_library(libsherpa-onnx-kaldifst-core SHARED IMPORTED)
set_target_properties(libsherpa-onnx-kaldifst-core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libsherpa-onnx-kaldifst-core.dylib)
add_library(libucd SHARED IMPORTED)
set_target_properties(libucd PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/sherpa-onnx-tts/sherpa-onnx/libucd.dylib)




add_subdirectory(yolov5-lite)
add_subdirectory(yolo-nas)
add_subdirectory(silero-vad)

add_executable(test_onnxruntime main.cpp)

target_link_libraries(test_onnxruntime onnxruntime)
#target_link_libraries(test_onnxruntime libcargs)
#target_link_libraries(test_onnxruntime libespeak-ng)
#target_link_libraries(test_onnxruntime libkaldi-decoder-core)
target_link_libraries(test_onnxruntime libpiper_phonemize.1.2.0)
#target_link_libraries(test_onnxruntime libpiper_phonemize.1)
#target_link_libraries(test_onnxruntime libpiper_phonemize)
target_link_libraries(test_onnxruntime libsherpa-onnx-c-api)
target_link_libraries(test_onnxruntime libsherpa-onnx-core)
target_link_libraries(test_onnxruntime libsherpa-onnx-fst.6)
#target_link_libraries(test_onnxruntime libsherpa-onnx-fst)
target_link_libraries(test_onnxruntime libsherpa-onnx-kaldifst-core)
target_link_libraries(test_onnxruntime libucd)
target_link_libraries(test_onnxruntime ${OpenCV_LIBS})
target_link_libraries(test_onnxruntime ${yolo_nas_lib})
target_link_libraries(test_onnxruntime ${yolov5_lite_lib})
target_link_libraries(test_onnxruntime ${sherpa_onnx_lib})
target_link_libraries(test_onnxruntime ${silero_vad_lib})


