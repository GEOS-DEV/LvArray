#
set( PREPROCESSOR_DEFINES ARRAY_BOUNDS_CHECK
                          ATK
                          CHAI
                          CUDA
                          MPI
                          RAJA
                          TOTALVIEW_OUTPUT
                          OPENMP
   )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} )
        set(USE_${DEP} TRUE  )
    endif()
endforeach()

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/LvArrayConfig.hpp )

function( make_full_config_file 
          PREPROCESSOR_VARS )
    foreach( DEP in ${PREPROCESSOR_VARS})
        set(USE_${DEP} TRUE  )
        set(GEOSX_USE_${DEP} TRUE  )
        set(${DEP} TRUE  )
    endforeach()

    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/docs/doxygen/LvArrayConfig.hpp )
endfunction()


make_full_config_file( "${PREPROCESSOR_DEFINES}" )

