# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/unist-escl/Desktop/realtime_image/Training

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/unist-escl/Desktop/realtime_image/Training/build

# Include any dependencies generated for this target.
include CMakeFiles/RT_NN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RT_NN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RT_NN.dir/flags.make

CMakeFiles/RT_NN.dir/src/main_training.o: CMakeFiles/RT_NN.dir/flags.make
CMakeFiles/RT_NN.dir/src/main_training.o: ../src/main_training.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unist-escl/Desktop/realtime_image/Training/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RT_NN.dir/src/main_training.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RT_NN.dir/src/main_training.o -c /home/unist-escl/Desktop/realtime_image/Training/src/main_training.cpp

CMakeFiles/RT_NN.dir/src/main_training.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RT_NN.dir/src/main_training.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unist-escl/Desktop/realtime_image/Training/src/main_training.cpp > CMakeFiles/RT_NN.dir/src/main_training.i

CMakeFiles/RT_NN.dir/src/main_training.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RT_NN.dir/src/main_training.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unist-escl/Desktop/realtime_image/Training/src/main_training.cpp -o CMakeFiles/RT_NN.dir/src/main_training.s

CMakeFiles/RT_NN.dir/src/main_training.o.requires:

.PHONY : CMakeFiles/RT_NN.dir/src/main_training.o.requires

CMakeFiles/RT_NN.dir/src/main_training.o.provides: CMakeFiles/RT_NN.dir/src/main_training.o.requires
	$(MAKE) -f CMakeFiles/RT_NN.dir/build.make CMakeFiles/RT_NN.dir/src/main_training.o.provides.build
.PHONY : CMakeFiles/RT_NN.dir/src/main_training.o.provides

CMakeFiles/RT_NN.dir/src/main_training.o.provides.build: CMakeFiles/RT_NN.dir/src/main_training.o


CMakeFiles/RT_NN.dir/src/modules/dataReader.o: CMakeFiles/RT_NN.dir/flags.make
CMakeFiles/RT_NN.dir/src/modules/dataReader.o: ../src/modules/dataReader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unist-escl/Desktop/realtime_image/Training/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/RT_NN.dir/src/modules/dataReader.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RT_NN.dir/src/modules/dataReader.o -c /home/unist-escl/Desktop/realtime_image/Training/src/modules/dataReader.cpp

CMakeFiles/RT_NN.dir/src/modules/dataReader.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RT_NN.dir/src/modules/dataReader.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unist-escl/Desktop/realtime_image/Training/src/modules/dataReader.cpp > CMakeFiles/RT_NN.dir/src/modules/dataReader.i

CMakeFiles/RT_NN.dir/src/modules/dataReader.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RT_NN.dir/src/modules/dataReader.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unist-escl/Desktop/realtime_image/Training/src/modules/dataReader.cpp -o CMakeFiles/RT_NN.dir/src/modules/dataReader.s

CMakeFiles/RT_NN.dir/src/modules/dataReader.o.requires:

.PHONY : CMakeFiles/RT_NN.dir/src/modules/dataReader.o.requires

CMakeFiles/RT_NN.dir/src/modules/dataReader.o.provides: CMakeFiles/RT_NN.dir/src/modules/dataReader.o.requires
	$(MAKE) -f CMakeFiles/RT_NN.dir/build.make CMakeFiles/RT_NN.dir/src/modules/dataReader.o.provides.build
.PHONY : CMakeFiles/RT_NN.dir/src/modules/dataReader.o.provides

CMakeFiles/RT_NN.dir/src/modules/dataReader.o.provides.build: CMakeFiles/RT_NN.dir/src/modules/dataReader.o


CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o: CMakeFiles/RT_NN.dir/flags.make
CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o: ../src/modules/neuralNetworkTrainer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unist-escl/Desktop/realtime_image/Training/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o -c /home/unist-escl/Desktop/realtime_image/Training/src/modules/neuralNetworkTrainer.cpp

CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unist-escl/Desktop/realtime_image/Training/src/modules/neuralNetworkTrainer.cpp > CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.i

CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unist-escl/Desktop/realtime_image/Training/src/modules/neuralNetworkTrainer.cpp -o CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.s

CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.requires:

.PHONY : CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.requires

CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.provides: CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.requires
	$(MAKE) -f CMakeFiles/RT_NN.dir/build.make CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.provides.build
.PHONY : CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.provides

CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.provides.build: CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o


CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o: CMakeFiles/RT_NN.dir/flags.make
CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o: ../src/modules/neuralNetwork.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unist-escl/Desktop/realtime_image/Training/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o -c /home/unist-escl/Desktop/realtime_image/Training/src/modules/neuralNetwork.cpp

CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unist-escl/Desktop/realtime_image/Training/src/modules/neuralNetwork.cpp > CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.i

CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unist-escl/Desktop/realtime_image/Training/src/modules/neuralNetwork.cpp -o CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.s

CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.requires:

.PHONY : CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.requires

CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.provides: CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.requires
	$(MAKE) -f CMakeFiles/RT_NN.dir/build.make CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.provides.build
.PHONY : CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.provides

CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.provides.build: CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o


# Object files for target RT_NN
RT_NN_OBJECTS = \
"CMakeFiles/RT_NN.dir/src/main_training.o" \
"CMakeFiles/RT_NN.dir/src/modules/dataReader.o" \
"CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o" \
"CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o"

# External object files for target RT_NN
RT_NN_EXTERNAL_OBJECTS =

../RT_NN: CMakeFiles/RT_NN.dir/src/main_training.o
../RT_NN: CMakeFiles/RT_NN.dir/src/modules/dataReader.o
../RT_NN: CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o
../RT_NN: CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o
../RT_NN: CMakeFiles/RT_NN.dir/build.make
../RT_NN: /usr/local/lib/libopencv_cudabgsegm.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudaobjdetect.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudastereo.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_stitching.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_superres.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_videostab.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_aruco.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_bgsegm.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_bioinspired.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_ccalib.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_dpm.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_face.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_fuzzy.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_hdf.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_img_hash.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_line_descriptor.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_optflow.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_reg.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_rgbd.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_saliency.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_sfm.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_stereo.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_structured_light.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_surface_matching.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_tracking.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_xfeatures2d.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_ximgproc.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_xobjdetect.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_xphoto.so.3.3.1
../RT_NN: /usr/lib/x86_64-linux-gnu/libcuda.so
../RT_NN: /usr/local/cuda-8.0/lib64/libcudart.so
../RT_NN: /usr/local/cuda-8.0/lib64/libnppc.so
../RT_NN: /usr/local/cuda-8.0/lib64/libnppi.so
../RT_NN: /usr/local/cuda-8.0/lib64/libnpps.so
../RT_NN: /usr/local/lib/libopencv_cudafeatures2d.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudacodec.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudaoptflow.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudalegacy.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudawarping.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_photo.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudaimgproc.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudafilters.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_datasets.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_plot.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_text.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_dnn.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudaarithm.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_ml.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_shape.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_video.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_calib3d.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_features2d.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_highgui.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_videoio.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_flann.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_imgcodecs.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_objdetect.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_imgproc.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_core.so.3.3.1
../RT_NN: /usr/local/lib/libopencv_cudev.so.3.3.1
../RT_NN: CMakeFiles/RT_NN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/unist-escl/Desktop/realtime_image/Training/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../RT_NN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RT_NN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RT_NN.dir/build: ../RT_NN

.PHONY : CMakeFiles/RT_NN.dir/build

CMakeFiles/RT_NN.dir/requires: CMakeFiles/RT_NN.dir/src/main_training.o.requires
CMakeFiles/RT_NN.dir/requires: CMakeFiles/RT_NN.dir/src/modules/dataReader.o.requires
CMakeFiles/RT_NN.dir/requires: CMakeFiles/RT_NN.dir/src/modules/neuralNetworkTrainer.o.requires
CMakeFiles/RT_NN.dir/requires: CMakeFiles/RT_NN.dir/src/modules/neuralNetwork.o.requires

.PHONY : CMakeFiles/RT_NN.dir/requires

CMakeFiles/RT_NN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RT_NN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RT_NN.dir/clean

CMakeFiles/RT_NN.dir/depend:
	cd /home/unist-escl/Desktop/realtime_image/Training/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/unist-escl/Desktop/realtime_image/Training /home/unist-escl/Desktop/realtime_image/Training /home/unist-escl/Desktop/realtime_image/Training/build /home/unist-escl/Desktop/realtime_image/Training/build /home/unist-escl/Desktop/realtime_image/Training/build/CMakeFiles/RT_NN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RT_NN.dir/depend

