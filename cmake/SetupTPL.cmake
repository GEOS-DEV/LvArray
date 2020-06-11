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


################################
# CALIPER and Adiak
################################
if(ENABLE_CALIPER)
    if(NOT EXISTS ${ADIAK_DIR})
        set(ADIAK_DIR ${GEOSX_TPL_DIR}/adiak)
    endif()

    message(STATUS "Using adiak at ${ADIAK_DIR}")

    find_package(adiak REQUIRED
                 PATHS ${ADIAK_DIR}/lib/cmake/adiak)

    blt_register_library(NAME adiak
                         INCLUDES ${adiak_INCLUDE_DIRS}
                         LIBRARIES ${adiak_LIBRARIES}
                         TREAT_INCLUDES_AS_SYSTEM ON)

    set(thirdPartyLibs ${thirdPartyLibs} adiak)

    if(NOT EXISTS ${CALIPER_DIR})
        set(CALIPER_DIR ${GEOSX_TPL_DIR}/caliper)
    endif()

    message(STATUS "Using caliper at ${CALIPER_DIR}")

    find_package(caliper REQUIRED
                 PATHS ${CALIPER_DIR}/share/cmake/caliper)
 
    if(ENABLE_MPI)
        set(caliper_LIBRARIES caliper-mpi)
    else()
        set(caliper_LIBRARIES caliper)
    endif()

    blt_register_library(NAME caliper
                         INCLUDES ${caliper_INCLUDE_PATH}
                         LIBRARIES ${caliper_LIBRARIES}
                         TREAT_INCLUDES_AS_SYSTEM ON)

    set(thirdPartyLibs ${thirdPartyLibs} caliper)
endif()




set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )
