###################################################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
# All rights reserved.
# See the LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###################################################################################################

cmake_minimum_required( VERSION 3.9 )

# Set version number
set( LVARRAY_VERSION_MAJOR 0 )
set( LVARRAY_VERSION_MINOR 1 )
set( LVARRAY_VERSION_PATCHLEVEL 0 )

# check if LvArray is build as a submodule or a separate project
get_directory_property( parent_dir PARENT_DIRECTORY )
if(parent_dir)
    set( is_submodule ON )
else()
    set( is_submodule OFF )
endif()

if( NOT is_submodule )

    project( LvArray LANGUAGES C CXX )

    set( BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard" FORCE )
    set( ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "" )

    option( ENABLE_TESTS "Builds tests" ON )
    option( DISABLE_UNIT_TESTS "Disables unit tests but builds gtest." OFF )
    option( ENABLE_EXAMPLES "Builds examples" ON )
    option( ENABLE_BENCHMARKS "Builds benchmarks" ON )
    option( ENABLE_DOCS "Builds documentation" ON )

    option( ENABLE_ADDR2LINE "Enable addr2line usage in stacktraces" ON )
    option( ENABLE_CUDA "Build with CUDA" OFF )
    option( ENABLE_UMPIRE "Build with UMPIRE" OFF )
    option( ENABLE_CHAI "Build with CHAI" OFF )
    option( ENABLE_CALIPER "Build with Caliper" OFF )

    include( cmake/blt/SetupBLT.cmake )
    include( cmake/CMakeBasics.cmake )
    include( cmake/SetupTPL.cmake )
else()
    include( cmake/CMakeBasics.cmake )
endif()

include(cmake/Macros.cmake)
include(cmake/Config.cmake)

set( lvarray_dependencies dl )

if( ENABLE_CHAI )
    set( lvarray_dependencies ${lvarray_dependencies} chai umpire )
endif()

if( ENABLE_CUDA )
    set( lvarray_dependencies ${lvarray_dependencies} cuda )
endif()

if( ENABLE_RAJA )
    set( lvarray_dependencies ${lvarray_dependencies} RAJA )
endif()

if( ENABLE_CALIPER )
    set( lvarray_dependencies ${lvarray_dependencies} caliper )
endif()

add_subdirectory( src )

if( ENABLE_TESTS AND NOT DISABLE_UNIT_TESTS )
    add_subdirectory( unitTests )
endif()

if ( ENABLE_EXAMPLES )
    if( NOT ENABLE_TESTS )
        message( WARNING "ENABLE_TESTS must be on to build the examples." )
    endif()

    add_subdirectory( examples )
endif()

if( ENABLE_BENCHMARKS )
    if( NOT ENABLE_TESTS )
        message( WARNING "ENABLE_TESTS must be on to build the benchmarks." )
    endif()

    add_subdirectory( benchmarks )
endif()

if( ENABLE_DOCS )
    add_subdirectory( docs )
endif()


