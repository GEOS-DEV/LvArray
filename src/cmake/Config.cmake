#
set( PREPROCESSOR_DEFINES 
                          ARRAY_BOUNDS_CHECK
                          AXOM
                          CHAI
                          MPI
   )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} )
        set(USE_${DEP} TRUE  )
    endif()
endforeach()

message( "Config.cmake: CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}" )
message( "Config.cmake: CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}" )

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/src/CXX_UtilsConfig.hpp.in
    ${CMAKE_BINARY_DIR}/include/CXX_UtilsConfig.hpp
)


