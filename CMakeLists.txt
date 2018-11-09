cmake_minimum_required(VERSION 3.9)



# check if cxx-utilities is build as a submodule or a separate project
get_directory_property(parent_dir PARENT_DIRECTORY)
if(parent_dir)
    set(is_submodule ON)
else()
    set(is_submodule OFF)
endif()

if(NOT is_submodule)

  project(cxx-utilities LANGUAGES CXX)


#  cmake_policy( SET CMP0042 NEW )
#  cmake_policy( SET CMP0074 NEW )

  set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard" FORCE)
  set(ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "")

  include(src/cmake/blt/SetupBLT.cmake)
  include(src/cmake/CMakeBasics.cmake)
  include(src/cmake/SetupTPL.cmake)

endif() 

include( src/cmake/Config.cmake )
add_subdirectory(src)
add_subdirectory(src/docs)

#install(DIRECTORY ${PROJECT_BINARY_DIR}/bin DESTINATION . OPTIONAL)
#install(DIRECTORY ${PROJECT_BINARY_DIR}/include DESTINATION . OPTIONAL)