#
set( PREPROCESSOR_DEFINES 
                          ARRAY_BOUNDS_CHECK
                          ATK
                          CHAI
                          CUDA
                          MPI
                          TOTALVIEW_OUTPUT
                          OPENMP
   )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} )
        set(USE_${DEP} TRUE  )
    endif()
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/src/CXX_UtilsConfig.hpp.in
    ${CMAKE_BINARY_DIR}/include/CXX_UtilsConfig.hpp
)


