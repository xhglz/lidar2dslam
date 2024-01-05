
add_executable(test_2dlidar_io test_2dlidar_io.cc)
target_link_libraries(test_2dlidar_io
    ${PROJECT_NAME}.slam
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    glog gflags
    ${PCL_LIBRARIES}
    ${third_party_libs}
)

add_executable(test_2d_icp_s2s test_2d_icp_s2s.cc)
target_link_libraries(test_2d_icp_s2s
    ${PROJECT_NAME}.slam
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    glog gflags
    ${PCL_LIBRARIES}
    ${third_party_libs}
)

add_executable(test_2d_icp_likelihood test_2d_icp_likelihood.cc)
target_link_libraries(test_2d_icp_likelihood
    ${PROJECT_NAME}.slam
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    glog gflags
    ${PCL_LIBRARIES}
    ${third_party_libs}
)

add_executable(test_occupancy_grid test_occupancy_grid.cc)
target_link_libraries(test_occupancy_grid
    ${PROJECT_NAME}.slam
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    glog gflags
    ${PCL_LIBRARIES}
    ${third_party_libs}
)

add_executable(test_2d_mapping test_2d_mapping.cc)
target_link_libraries(test_2d_mapping
    ${PROJECT_NAME}.slam
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    glog gflags
    ${PCL_LIBRARIES}
    ${third_party_libs}
)

add_executable(test_mr_matching test_mr_matching.cc)
target_link_libraries(test_mr_matching
    ${PROJECT_NAME}.slam
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    glog gflags
    ${PCL_LIBRARIES}
    ${third_party_libs}
)
