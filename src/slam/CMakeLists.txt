list(APPEND libraries
    ${PROJECT_NAME}.ui
    ${PROJECT_NAME}.basic
    ${third_party_libs}
)

list(APPEND LIBRARY_SOURCES
    icp_2d.cc
    lidar_2d_utils.cc
    likelihood_field.cc
    occupancy_map.cc
    submap.cc
    mapping_2d.cc
    multi_resolution_likelihood_field.cc
    loop_closing.cc
    frame.cc
)
file(GLOB_RECURSE LIBRARY_HEADERS "core/*.h" "iekf/*.h")

add_library(${PROJECT_NAME}.slam ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(${PROJECT_NAME}.slam ${libraries})