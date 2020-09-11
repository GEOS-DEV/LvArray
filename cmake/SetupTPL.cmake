set(thirdPartyLibs "")

################################
# RAJA
################################
if(NOT EXISTS ${RAJA_DIR})
    set(RAJA_DIR ${GEOSX_TPL_DIR}/raja)
endif()

message(STATUS "Using RAJA from ${RAJA_DIR}")

find_package(RAJA REQUIRED PATHS ${RAJA_DIR})

set(ENABLE_RAJA ON CACHE BOOL "")

set(thirdPartyLibs ${thirdPartyLibs} RAJA)


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
                          PROPERTIES INTERFACE_LINK_LIBRARIES "${CHAI_LINK_LIBRARIES}")

    set(thirdPartyLibs ${thirdPartyLibs} chai)
else()
    message(STATUS "Not using chai.")
endif()


################################
# CALIPER
################################
if(ENABLE_CALIPER)
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
else()
    message(STATUS "Not using caliper.")
endif()


set(thirdPartyLibs ${thirdPartyLibs} CACHE STRING "")
