#
set( PREPROCESSOR_DEFINES ARRAY_BOUNDS_CHECK
                          CHAI
                          CUDA
                          MPI
                          TOTALVIEW_OUTPUT
                          OPENMP
                          CALIPER )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} )
        set( LVARRAY_USE_${DEP} TRUE )
    endif()
endforeach()

if( USE_ADDR2LINE )
    if ( NOT DEFINED ADDR2LINE_EXEC )
        set( ADDR2LINE_EXEC /usr/bin/addr2line CACHE PATH "" )
    endif()

    if ( NOT EXISTS ${ADDR2LINE_EXEC} )
        message( FATAL_ERROR "The addr2line executable does not exist: ${ADDR2LINE_EXEC}" )
    endif()

    set( LVARRAY_ADDR2LINE_EXEC ${ADDR2LINE_EXEC} )
endif()

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/LvArrayConfig.hpp )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/LvArrayConfig.hpp )
