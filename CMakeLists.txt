cmake_minimum_required(VERSION 3.10)

project(ModelDeploy)

# set a value to restore thirdParty directory
set(ONNXRUNTIME_DIR "/home/hgm/ThirdParty/onnxruntime-linux-x64-1.12.1")

include_directories("${ONNXRUNTIME_DIR}/include")

find_package(OpenCV 4 REQUIRED)

# 头文件目录
include_directories( ${OpenCV_INCLUDE_DIRS}/include ${CMAKE_SOURCE_DIR}/cv/include ${CMAKE_SOURCE_DIR}/ort/include)
include_directories( ${CMAKE_SOURCE_DIR})

# 源代码目录
aux_source_directory(${CMAKE_SOURCE_DIR}/cv/src SRC_CV )
aux_source_directory(${CMAKE_SOURCE_DIR}/ort/src SRC_ORT)
aux_source_directory(. SRC_CUR)


add_executable(ModelDeploy ${SRC_CV} ${SRC_ORT} ${SRC_CUR})

target_link_libraries(ModelDeploy PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS})