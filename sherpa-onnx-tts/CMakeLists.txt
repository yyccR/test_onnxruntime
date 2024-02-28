aux_source_directory(. sherpa)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../libs/opencv-mobile-4.5.4-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)

add_library(sherpa_onnx_lib ${sherpa})
target_link_libraries(sherpa_onnx_lib ${OpenCV_LIBS})
target_link_libraries(sherpa_onnx_lib onnxruntime)


