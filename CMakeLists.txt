cmake_minimum_required(VERSION 3.4)

################################
# CXX-utilities
################################
project(cxx-utilities LANGUAGES CXX)

set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard" FORCE)

################################
# Include blt
################################
include(src/cmake/blt/SetupBLT.cmake)

################################
# Include standard build system logic and options
################################
include(src/cmake/CMakeBasics.cmake)

################################
# Add components
################################
add_subdirectory(src)


################################
# Add docs
################################
#add_subdirectory(src/docs)


#add_code_check_targets(uncrustify.cfg)


################################
# install targets from build
################################
install(DIRECTORY ${PROJECT_BINARY_DIR}/bin DESTINATION . OPTIONAL)
install(DIRECTORY ${PROJECT_BINARY_DIR}/include DESTINATION . OPTIONAL)