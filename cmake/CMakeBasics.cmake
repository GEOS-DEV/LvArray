set(CMAKE_ENABLE_EXPORTS ON)

message( "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(ENABLE_ARRAY_BOUNDS_CHECK "ON" CACHE BOOL "")
endif()
option( ENABLE_ARRAY_BOUNDS_CHECK "" OFF )

option( ENABLE_TOTALVIEW_OUTPUT "" OFF )


if( ( NOT BLT_CXX_STD STREQUAL c++14 ) AND (NOT BLT_CXX_STD STREQUAL c++11))
    MESSAGE(FATAL_ERROR "c++11/14 is NOT enabled. LvArray requires c++11/14")
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
