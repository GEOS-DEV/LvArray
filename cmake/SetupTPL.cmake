set(thirdPartyLibs "")

################################
# CAMP
################################
if(NOT EXISTS ${CAMP_DIR})
    message(FATAL_ERROR "CAMP_DIR must be defined and point to a valid directory when using CAMP.")
endif()

message(STATUS "Using CAMP from ${CAMP_DIR}")

find_package(camp REQUIRED PATHS ${CAMP_DIR})

set(ENABLE_CAMP ON CACHE BOOL "")

set(thirdPartyLibs ${thirdPartyLibs} camp)

################################
# RAJA
################################
if(NOT EXISTS ${RAJA_DIR})
    message(FATAL_ERROR "RAJA_DIR must be defined and point to a valid directory when using RAJA.")
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
        message(FATAL_ERROR "UMPIRE_DIR must be defined and point to a valid directory when using Umpire.")
    endif()

    message(STATUS "Using Umpire from ${UMPIRE_DIR}")

    find_package(umpire REQUIRED
                 PATHS ${UMPIRE_DIR})
    
    set(thirdPartyLibs ${thirdPartyLibs} umpire)
else()
    message(STATUS "Not using Umpire.")
endif()

################################
# CHAI
################################
if(ENABLE_CHAI)
    if(NOT ENABLE_UMPIRE)
        message(FATAL_ERROR "Umpire must be enabled to use CHAI.")
    endif()

    if(NOT ENABLE_RAJA)
        message(FATAL_ERROR "RAJA must be enabled to use CHAI.")
    endif()

    if(NOT EXISTS ${CHAI_DIR})
        message(FATAL_ERROR "CHAI_DIR must be defined and point to a valid directory when using CHAI.")
    endif()

    message(STATUS "Using CHAI from ${CHAI_DIR}")

    find_package(chai REQUIRED
                 PATHS ${CHAI_DIR})

    # If this isn't done chai will add -lRAJA to the link line, but we don't link to RAJA like that.
    get_target_property(CHAI_LINK_LIBRARIES chai INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM CHAI_LINK_LIBRARIES RAJA)
    set_target_properties(chai
                          PROPERTIES INTERFACE_LINK_LIBRARIES "${CHAI_LINK_LIBRARIES}")

    set(thirdPartyLibs ${thirdPartyLibs} chai)
else()
    message(STATUS "Not using CHAI.")
endif()


################################
# CALIPER
################################
if(ENABLE_CALIPER)
    if(NOT EXISTS ${CALIPER_DIR})
        message(FATAL_ERROR "CALIPER_DIR must be defined and point to a valid directory when using caliper.")
    endif()

    message(STATUS "Using caliper from ${CALIPER_DIR}")

    find_package(caliper REQUIRED
                 PATHS ${CALIPER_DIR})

    blt_register_library(NAME caliper
                         INCLUDES ${caliper_INCLUDE_PATH}
                         LIBRARIES caliper
                         TREAT_INCLUDES_AS_SYSTEM ON)

    set(thirdPartyLibs ${thirdPartyLibs} caliper)
else()
    message(STATUS "Not using caliper.")
endif()

################################
# Python
################################
if ( ENABLE_PYLVARRAY )
    message( STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}" )
    find_package( Python3 REQUIRED
                  COMPONENTS Development NumPy )

    message( STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}" )
    message( STATUS "Python3_LIBRARY_DIRS = ${Python3_LIBRARY_DIRS}" )
    message( STATUS "Python3_NumPy_INCLUDE_DIRS = ${Python3_NumPy_INCLUDE_DIRS}" )

    set( thirdPartyLibs ${thirdPartyLibs} Python3::Python Python3::NumPy )
endif()

set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )
