# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/albusgive/mujoco_learning/extend/touch/C++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/albusgive/mujoco_learning/extend/touch/C++/b

# Include any dependencies generated for this target.
include CMakeFiles/touchpad.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/touchpad.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/touchpad.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/touchpad.dir/flags.make

CMakeFiles/touchpad.dir/touchpad.cpp.o: CMakeFiles/touchpad.dir/flags.make
CMakeFiles/touchpad.dir/touchpad.cpp.o: ../touchpad.cpp
CMakeFiles/touchpad.dir/touchpad.cpp.o: CMakeFiles/touchpad.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/albusgive/mujoco_learning/extend/touch/C++/b/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/touchpad.dir/touchpad.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/touchpad.dir/touchpad.cpp.o -MF CMakeFiles/touchpad.dir/touchpad.cpp.o.d -o CMakeFiles/touchpad.dir/touchpad.cpp.o -c /home/albusgive/mujoco_learning/extend/touch/C++/touchpad.cpp

CMakeFiles/touchpad.dir/touchpad.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/touchpad.dir/touchpad.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/albusgive/mujoco_learning/extend/touch/C++/touchpad.cpp > CMakeFiles/touchpad.dir/touchpad.cpp.i

CMakeFiles/touchpad.dir/touchpad.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/touchpad.dir/touchpad.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/albusgive/mujoco_learning/extend/touch/C++/touchpad.cpp -o CMakeFiles/touchpad.dir/touchpad.cpp.s

CMakeFiles/touchpad.dir/mujoco_thread.cpp.o: CMakeFiles/touchpad.dir/flags.make
CMakeFiles/touchpad.dir/mujoco_thread.cpp.o: ../mujoco_thread.cpp
CMakeFiles/touchpad.dir/mujoco_thread.cpp.o: CMakeFiles/touchpad.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/albusgive/mujoco_learning/extend/touch/C++/b/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/touchpad.dir/mujoco_thread.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/touchpad.dir/mujoco_thread.cpp.o -MF CMakeFiles/touchpad.dir/mujoco_thread.cpp.o.d -o CMakeFiles/touchpad.dir/mujoco_thread.cpp.o -c /home/albusgive/mujoco_learning/extend/touch/C++/mujoco_thread.cpp

CMakeFiles/touchpad.dir/mujoco_thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/touchpad.dir/mujoco_thread.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/albusgive/mujoco_learning/extend/touch/C++/mujoco_thread.cpp > CMakeFiles/touchpad.dir/mujoco_thread.cpp.i

CMakeFiles/touchpad.dir/mujoco_thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/touchpad.dir/mujoco_thread.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/albusgive/mujoco_learning/extend/touch/C++/mujoco_thread.cpp -o CMakeFiles/touchpad.dir/mujoco_thread.cpp.s

# Object files for target touchpad
touchpad_OBJECTS = \
"CMakeFiles/touchpad.dir/touchpad.cpp.o" \
"CMakeFiles/touchpad.dir/mujoco_thread.cpp.o"

# External object files for target touchpad
touchpad_EXTERNAL_OBJECTS =

touchpad: CMakeFiles/touchpad.dir/touchpad.cpp.o
touchpad: CMakeFiles/touchpad.dir/mujoco_thread.cpp.o
touchpad: CMakeFiles/touchpad.dir/build.make
touchpad: /opt/mujoco/lib/libmujoco.so.3.3.4
touchpad: /usr/local/lib/libopencv_gapi.so.4.10.0
touchpad: /usr/local/lib/libopencv_highgui.so.4.10.0
touchpad: /usr/local/lib/libopencv_ml.so.4.10.0
touchpad: /usr/local/lib/libopencv_objdetect.so.4.10.0
touchpad: /usr/local/lib/libopencv_photo.so.4.10.0
touchpad: /usr/local/lib/libopencv_stitching.so.4.10.0
touchpad: /usr/local/lib/libopencv_video.so.4.10.0
touchpad: /usr/local/lib/libopencv_videoio.so.4.10.0
touchpad: /usr/local/lib/libopencv_imgcodecs.so.4.10.0
touchpad: /usr/local/lib/libopencv_dnn.so.4.10.0
touchpad: /usr/local/lib/libopencv_calib3d.so.4.10.0
touchpad: /usr/local/lib/libopencv_features2d.so.4.10.0
touchpad: /usr/local/lib/libopencv_flann.so.4.10.0
touchpad: /usr/local/lib/libopencv_imgproc.so.4.10.0
touchpad: /usr/local/lib/libopencv_core.so.4.10.0
touchpad: CMakeFiles/touchpad.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/albusgive/mujoco_learning/extend/touch/C++/b/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable touchpad"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/touchpad.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/touchpad.dir/build: touchpad
.PHONY : CMakeFiles/touchpad.dir/build

CMakeFiles/touchpad.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/touchpad.dir/cmake_clean.cmake
.PHONY : CMakeFiles/touchpad.dir/clean

CMakeFiles/touchpad.dir/depend:
	cd /home/albusgive/mujoco_learning/extend/touch/C++/b && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/albusgive/mujoco_learning/extend/touch/C++ /home/albusgive/mujoco_learning/extend/touch/C++ /home/albusgive/mujoco_learning/extend/touch/C++/b /home/albusgive/mujoco_learning/extend/touch/C++/b /home/albusgive/mujoco_learning/extend/touch/C++/b/CMakeFiles/touchpad.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/touchpad.dir/depend

