set( thirdPartyLibs "")


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
blt_register_library( NAME RAJA
                      INCLUDES ${RAJA_INCLUDE_DIRS}
                      LIBRARIES ${RAJA_LIBRARY}
                      TREAT_INCLUDES_AS_SYSTEM ON )

set(ENABLE_RAJA ON CACHE BOOL "")

set( thirdPartyLibs ${thirdPartyLibs} raja )


###############################
# UMPIRE
###############################
if(ENABLE_UMPIRE)
    if(NOT EXISTS ${UMPIRE_DIR})
        set(UMPIRE_DIR ${GEOSX_TPL_DIR}/chai)
    endif()

    message(STATUS "Using umpire at ${UMPIRE_DIR}")

    find_package(umpire REQUIRED
                 PATHS ${UMPIRE_DIR}/share/umpire/cmake/)
    
    set(thirdPartyLibs ${thirdPartyLibs} umpire)
else()
    message(STATUS "Not using umpire.")
endif()

################################
# CHAI
################################
# include(cmake/FindCHAI.cmake)
if(ENABLE_CHAI)
    if(NOT ENABLE_UMPIRE)
        message(FATAL_ERROR "umpire must be enabled to use chai.")
    endif()

    if(NOT ENABLE_RAJA)
        message(FATAL_ERROR "RAJA must be enabled to use chai.")
    endif()

    if(NOT EXISTS ${CHAI_DIR})
        set(CHAI_DIR ${GEOSX_TPL_DIR}/chai)
    endif()

    message(STATUS "Using chai at ${CHAI_DIR}")

    find_package(chai REQUIRED
                 PATHS ${CHAI_DIR}/share/chai/cmake/)

    # If this isn't done chai will add -lRAJA to the link line, but we don't link to RAJA like that.
    get_target_property(CHAI_LINK_LIBRARIES chai INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM CHAI_LINK_LIBRARIES RAJA)
    set_target_properties(chai
                        PROPERTIES INTERFACE_LINK_LIBRARIES ${CHAI_LINK_LIBRARIES})

    set(thirdPartyLibs ${thirdPartyLibs} chai)
else()
    message(STATUS "Not using chai.")
endif()


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

################################
# Python
################################
if(ENABLE_PYTHON)
    include(cmake/FindPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
endif()




set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )
