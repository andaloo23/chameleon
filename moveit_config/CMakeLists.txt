cmake_minimum_required(VERSION 3.8)
project(so100_moveit_config)

find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY
  config
  launch
  urdf
  assets
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
