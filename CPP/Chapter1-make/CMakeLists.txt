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

#编译版本不从opt中使用
# set(MUJOCO_PATH "/home/albusgive/software/mujoco-3.3.2")
# include_directories(${MUJOCO_PATH}/include)
# link_directories(${MUJOCO_PATH}/build/bin)
# set(MUJOCO_LIB ${MUJOCO_PATH}/build/lib/libmujoco.so)

#release版本
# set(MUJOCO_PATH "/home/albusgive/software/mujoco-3.3.1-linux-x86_64/mujoco-3.3.1")
# include_directories(${MUJOCO_PATH}/include)
# link_directories(${MUJOCO_PATH}/bin)
# set(MUJOCO_LIB ${MUJOCO_PATH}/lib/libmujoco.so)


add_executable(basic basic.cc)
#从cmake安装位置opt使用
target_link_libraries(basic mujoco::mujoco glut GL GLU glfw)
#后两个使用
# target_link_libraries(basic ${MUJOCO_LIB} glut GL GLU glfw)


#simulate编译
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)
# file(GLOB SIMULATE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/simulate/*.cc)
# add_executable(simulate ${SIMULATE_SRC})
# target_link_libraries(simulate mujoco::mujoco glut GL GLU glfw lodepng)