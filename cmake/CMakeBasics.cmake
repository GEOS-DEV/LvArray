set(CMAKE_ENABLE_EXPORTS ON)

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  option( LVARRAY_BOUNDS_CHECK "" ON )
else()
  option( LVARRAY_BOUNDS_CHECK "" OFF )
endif()


option( ENABLE_TOTALVIEW_OUTPUT "" OFF )

option( LvArrayBuildAsObj "" OFF )


if( NOT BLT_CXX_STD STREQUAL c++14 )
    MESSAGE(FATAL_ERROR "c++14 is NOT enabled. LvArray requires c++14")
endif()


blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT "${OpenMP_CXX_FLAGS}")
blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS
                                 GNU   "-Wall -Wextra -Wpedantic -pedantic-errors -Wshadow -Wfloat-equal -Wcast-align -Wcast-qual"
                                 CLANG "-Wall -Wextra -Wpedantic -pedantic-errors -Wshadow -Wfloat-equal -Wcast-align -Wcast-qual"
                               )

blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS_DEBUG
                                 GNU ""
                                 CLANG "-fstandalone-debug"
                                )

blt_append_custom_compiler_flag(FLAGS_VAR GEOSX_NINJA_FLAGS
                  DEFAULT     " "
                  GNU         "-fdiagnostics-color=always"
                  CLANG       "-fcolor-diagnostics"
                  )

if( ${CMAKE_MAKE_PROGRAM} STREQUAL "ninja" OR ${CMAKE_MAKE_PROGRAM} MATCHES ".*/ninja$" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GEOSX_NINJA_FLAGS}")
endif()

