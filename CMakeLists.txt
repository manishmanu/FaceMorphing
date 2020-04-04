cmake_minimum_required(VERSION 2.8.9)
project (FaceMorphing)

find_package( OpenCV REQUIRED )

include_directories(inc)

set(
	SRC_FILES
	face_landmark_detection.cc
	face_morphing.cc
	subdiv2D_index.cc
)

add_library(face_morphing STATIC ${SRC_FILES})
target_link_libraries(face_morphing ${OpenCV_LIBS})

add_executable(example examples/example.cc)
target_link_libraries(example face_morphing ${OpenCV_LIBS})