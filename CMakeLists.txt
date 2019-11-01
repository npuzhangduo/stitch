# cmake needs this line 
cmake_minimum_required(VERSION 2.8) 
# Enable C++11 
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED TRUE) 
# Define project name 
project(stitch) 
include_directories(
include
)
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

SET(TEST_MATH
${DIR_SRCS}
)
# Find OpenCV, you may need to set OpenCV_DIR variable 
# to the absolute path to the directory containing OpenCVConfig.cmake file 
# via the command line or GUI 
find_package(OpenCV REQUIRED) 
include_directories( ${OpenCV_INCLUDE_DIRS} ) 
# Declare the executable target built from your sources 
link_directories("/home/zhangduo/anaconda3/lib")  
add_executable(main src/main.cpp src/myStitch.cpp) # Link your application with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})

