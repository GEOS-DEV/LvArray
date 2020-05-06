macro(lvarray_add_code_checks)

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
endmacro(lvarray_add_code_checks)