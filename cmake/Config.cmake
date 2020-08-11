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
        set(USE_${DEP} TRUE  )
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

function( make_full_config_file 
          PREPROCESSOR_VARS )
    foreach( DEP in ${PREPROCESSOR_VARS})
        set(USE_${DEP} TRUE )
        set(GEOSX_USE_${DEP} TRUE )
        set(${DEP} TRUE )
    endforeach()

    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                    ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/LvArrayConfig.hpp )
endfunction()


make_full_config_file( "${PREPROCESSOR_DEFINES}" )

