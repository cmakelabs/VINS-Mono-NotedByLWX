cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# for gdb debug 2024-7-13.
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# the end.

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    visualization_msgs
    )

# find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
find_package(fmt REQUIRED) # 2024-7-11.
add_definitions(-DSPDLOG_FMT_EXTERNAL) # 2024-7-11

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  /root/SETUP/symforce_install/include # added by wxliu on 2024-7-9
  # /root/SETUP/symforce_install/include/spdlog # 2024-7-11
)

link_directories(/root/SETUP/symforce_install/lib) # 2024-7-9

catkin_package()

add_executable(vins_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    )


target_link_libraries(vins_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}
    symforce_gen # 2024-7-9
    symforce_opt # 2024-7-9
    fmt::fmt # 2024-7-11
  ) 


