cmake_minimum_required(VERSION 3.9)

# check if LvArray is build as a submodule or a separate project
get_directory_property(parent_dir PARENT_DIRECTORY)
if(parent_dir)
    set(is_submodule ON)
else()
    set(is_submodule OFF)
endif()

if(NOT is_submodule)

  project(LvArray LANGUAGES C CXX)

  set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard" FORCE)
  set(ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "")

  option(ENABLE_BENCHMARKS "Enables benchmarks" ON)
  include(cmake/blt/SetupBLT.cmake)
  include(cmake/CMakeBasics.cmake)
  include(cmake/SetupTPL.cmake)

endif()

include(cmake/Macros.cmake)
include(cmake/Config.cmake)

set( lvarray_dependencies "" )

if( ENABLE_CHAI )
    set( lvarray_dependencies ${lvarray_dependencies} chai umpire )
endif()

if( ${ENABLE_MPI} )
    set( lvarray_dependencies ${lvarray_dependencies} mpi )
endif()

if( ENABLE_CUDA )
    set( lvarray_dependencies ${lvarray_dependencies} cuda )
endif()

if( ENABLE_OPENMP )
    set( lvarray_dependencies ${lvarray_dependencies} openmp )
endif()

if( ENABLE_RAJA )
    set( lvarray_dependencies ${lvarray_dependencies} raja )
endif()

if( ENABLE_CALIPER )
    set( lvarray_dependencies ${lvarray_dependencies} caliper )
endif()

add_subdirectory(unitTests)
add_subdirectory(benchmarks)
add_subdirectory(src)
add_subdirectory(docs)

lvarray_add_code_checks(PREFIX lvarray
                         EXCLUDES "blt/*" )
