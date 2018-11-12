



message( "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
if( CMAKE_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)" )
  set(ENABLE_ARRAY_BOUNDS_CHECK "ON" CACHE BOOL "")
endif()
option( ENABLE_ARRAY_BOUNDS_CHECK "" OFF )


if( ( NOT BLT_CXX_STD STREQUAL c++14 ) AND (NOT BLT_CXX_STD STREQUAL c++11))
    MESSAGE(FATAL_ERROR "c++11/14 is NOT enabled. cxx-utilities requires c++11/14")
endif()


blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT "${OpenMP_CXX_FLAGS}")
blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS
                                 GNU "-pedantic-errors -Wno-abi  -Wshadow -Wfloat-equal -Wcast-align  -Wpointer-arith -Wwrite-strings -Wcast-qual -Wswitch-default  -Wno-vla  -Wno-switch-default  -Wno-unused-parameter  -Wno-unused-variable  -Wno-unused-function" 
                                 CLANG "-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-missing-prototypes -Wno-covered-switch-default -Wno-double-promotion -Wno-documentation -Wno-switch-enum -Wno-sign-conversion -Wno-unused-parameter -Wno-unused-variable -Wno-reserved-id-macro -Wno-weak-vtables -Wno-undefined-func-template -Wno-global-constructors" 
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


macro(cxxutils_add_code_checks)

    set(options)
    set(singleValueArgs PREFIX UNCRUSTIFY_CFG_FILE )
    set(multiValueArgs EXCLUDES )

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    set(_all_sources)
    file(GLOB_RECURSE _all_sources
         "*.cpp" "*.hpp" "*.cxx" "*.hxx" "*.cc" "*.c" "*.h" "*.hh"
         "*.F" "*.f" "*.f90" "*.F90")

    # Check for excludes
    if (NOT DEFINED arg_EXCLUDES)
        set(_sources ${_all_sources})
    else()
        set(_sources)
        foreach(_source ${_all_sources})
            set(_to_be_excluded FALSE)
            foreach(_exclude ${arg_EXCLUDES})
                if (${_source} MATCHES ${_exclude})
                    set(_to_be_excluded TRUE)
                    break()
                endif()
            endforeach()

            if (NOT ${_to_be_excluded})
                list(APPEND _sources ${_source})
            endif()
        endforeach()
    endif()

    blt_add_code_checks( PREFIX    ${arg_PREFIX}
                         SOURCES   ${_sources}
                         UNCRUSTIFY_CFG_FILE ${PROJECT_SOURCE_DIR}/uncrustify.cfg
                       )
endmacro(cxxutils_add_code_checks)
                       