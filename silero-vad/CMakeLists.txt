aux_source_directory(. silero_vad)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../libs/opencv-mobile-4.5.4-android/sdk/native/jni)
#find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(silero_vad_lib ${silero_vad})
#target_link_libraries(sherpa_onnx_lib ${OpenCV_LIBS})
target_link_libraries(silero_vad_lib onnxruntime)


