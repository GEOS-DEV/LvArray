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

  set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard" FORCE)
  set(ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "")

  include(cmake/blt/SetupBLT.cmake)
  include(cmake/CMakeBasics.cmake)
  include(cmake/SetupTPL.cmake)

endif()

include(cmake/Macros.cmake)
include(cmake/Config.cmake)

set( cxx_dependencies "" )
if( ATK_FOUND )
    set( cxx_dependencies axom )
endif()

if( ENABLE_CHAI )
    set( cxx_dependencies ${cxx_dependencies} chai umpire )
endif()

if( ${ENABLE_MPI} )
    set( cxx_dependencies ${cxx_dependencies} mpi )
endif()

if( ENABLE_CUDA )
    set( cxx_dependencies ${cxx_dependencies} cuda )
endif()

if( ENABLE_OPENMP )
    set( cxx_dependencies ${cxx_dependencies} openmp )
endif()

if( ENABLE_RAJA )
    set( cxx_dependencies ${cxx_dependencies} raja )
endif()

if( ENABLE_CALIPER )
    set( cxx_dependencies ${cxx_dependencies} caliper )
endif()

add_subdirectory(unitTests)
add_subdirectory(src)
add_subdirectory(docs)

#install(DIRECTORY ${PROJECT_BINARY_DIR}/bin DESTINATION . OPTIONAL)
#install(DIRECTORY ${PROJECT_BINARY_DIR}/include DESTINATION . OPTIONAL)
