set( thirdPartyLibs "")



################################
# CHAI
################################
include(${CMAKE_SOURCE_DIR}/cmake/FindCHAI.cmake)


################################
# RAJA
################################
if( EXISTS ${RAJA_DIR})
    message(STATUS "Using system RAJA found at ${RAJA_DIR}")
else()
    message(STATUS "Using RAJA from thirdPartyLibs")
    set(RAJA_DIR ${GEOSX_TPL_DIR}/raja)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/FindRAJA.cmake)
if (NOT RAJA_FOUND)
    message(FATAL_ERROR "RAJA not found in ${RAJA_DIR}. Maybe you need to build it")
endif()    
blt_register_library( NAME raja
                      INCLUDES ${RAJA_INCLUDE_DIRS}
                      LIBRARIES ${RAJA_LIBRARY}
                      TREAT_INCLUDES_AS_SYSTEM ON )

set(ENABLE_RAJA ON CACHE BOOL "")

set( thirdPartyLibs ${thirdPartyLibs} raja )

set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )
