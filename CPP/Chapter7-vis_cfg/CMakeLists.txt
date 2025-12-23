cmake_minimum_required(VERSION 3.20)
project(MUJOCO_T)
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)

#编译安装，从cmake安装位置opt使用
set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
if (mujoco_FOUND)
    message(STATUS "Find mujoco:  ${mujoco_INCLUDE_DIRS}, ${mujoco_LIBRARIES}")
endif()

find_package(OpenCV REQUIRED)

add_executable(vis_cfg vis_cfg.cpp )
target_link_libraries(vis_cfg mujoco::mujoco glut GL GLU glfw ${OpenCV_LIBS})

