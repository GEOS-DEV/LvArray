#
set( PREPROCESSOR_DEFINES UMPIRE
                          CHAI
                          CUDA
			  HIP
                          TOTALVIEW_OUTPUT
                          CALIPER )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} )
        set( LVARRAY_USE_${DEP} TRUE )
    endif()
endforeach()

if( ENABLE_ADDR2LINE )
    if ( NOT DEFINED ADDR2LINE_EXEC )
        set( ADDR2LINE_EXEC /usr/bin/addr2line CACHE PATH "" )
    endif()

    if ( NOT EXISTS ${ADDR2LINE_EXEC} )
        message( FATAL_ERROR "The addr2line executable does not exist: ${ADDR2LINE_EXEC}" )
    endif()

    set( LVARRAY_ADDR2LINE_EXEC ${ADDR2LINE_EXEC} )
endif()

if( LVARRAY_BOUNDS_CHECK )
    message( STATUS "LvArray bounds checking enabled." )
else()
    message( STATUS "LvArray bounds checking disabled." )
endif()

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/LvArrayConfig.hpp )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/LvArrayConfig.hpp.in
                ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/LvArrayConfig.hpp )

# Install the generated header.
install( FILES ${CMAKE_BINARY_DIR}/include/LvArrayConfig.hpp
         DESTINATION include )

# Configure and install the CMake config
configure_file( ${CMAKE_CURRENT_LIST_DIR}/lvarray-config.cmake.in
                ${PROJECT_BINARY_DIR}/share/lvarray/cmake/lvarray-config.cmake)

install( FILES ${PROJECT_BINARY_DIR}/share/lvarray/cmake/lvarray-config.cmake
         DESTINATION share/lvarray/cmake/)
