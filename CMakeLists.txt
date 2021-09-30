cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(ALSEdu)

# find_package(OpenMP REQUIRED)
# if(OpenMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

find_package(OpenCV REQUIRED)

# include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include" "/usr/local/include" "/usr/include/python3.6")
# link_directories(${OpenCV_LIBRARY_DIRS} "/usr/lib/python3.6")
# add_definitions(${OpenCV_DEFINITIONS})
include_directories(${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include" "/usr/local/include")
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

set(CMAKE_BUILD_TYPE "Release")

add_executable(ALSTest src/ALSTest.cpp)
target_link_libraries(ALSTest ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(RobotSim src/RobotSim.cpp)
target_link_libraries(RobotSim ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(MCL src/MCL.cpp)
target_link_libraries(MCL ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(MCLRE src/MCLRE.cpp)
target_link_libraries(MCLRE ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(REsDatasetGeneration src/REsDatasetGeneration.cpp)
target_link_libraries(REsDatasetGeneration ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(MAEClassifierLearning src/MAEClassifierLearning.cpp)
target_link_libraries(MAEClassifierLearning yaml-cpp)

add_executable(AdaBoostClassifierLearning src/AdaBoostClassifierLearning.cpp)
target_link_libraries(AdaBoostClassifierLearning ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(MLPClassifierLearning src/MLPClassifierLearning.cpp)
target_link_libraries(MLPClassifierLearning ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(FailureDetection src/FailureDetection.cpp)
target_link_libraries(FailureDetection ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(E2EFusion src/E2EFusion.cpp)
target_link_libraries(E2EFusion ${OpenCV_LIBRARIES} yaml-cpp)

