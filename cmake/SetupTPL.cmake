set( thirdPartyLibs "")

################################
# AXOM
################################
if (NOT EXISTS ${ATK_DIR})
    message(INFO ": Using axom from thirdPartyLibs")
    set(ATK_DIR "${GEOSX_TPL_DIR}/axom" CACHE PATH "")
endif()

if (EXISTS ${ATK_DIR})
    message(INFO ": Using axom found at ${ATK_DIR}")
    
    set(ATK_FOUND TRUE)
    set(ATK_INCLUDE_DIRS ${ATK_DIR}/include)
    set(ATK_CMAKE ${ATK_DIR}/lib/cmake)

    include(${ATK_CMAKE}/core-targets.cmake)
    include(${ATK_CMAKE}/slic-targets.cmake)

    blt_register_library( NAME slic
                          INCLUDES ${ATK_INCLUDE_DIRS} 
                          LIBRARIES  slic
                          TREAT_INCLUDES_AS_SYSTEM ON)

    if( ENABLE_MPI )
      include(${ATK_CMAKE}/lumberjack-targets.cmake)
      blt_register_library( NAME lumberjack
                            INCLUDES ${ATK_INCLUDE_DIRS} 
                            LIBRARIES  lumberjack
                            TREAT_INCLUDES_AS_SYSTEM ON)

      set(thirdPartyLibs ${thirdPartyLibs} slic lumberjack )
    else()
      set(thirdPartyLibs ${thirdPartyLibs} slic )
    endif()
else()
    set(ATK_FOUND FALSE)
    message(INFO ": Not using axom")
endif()

################################
# CHAI
################################
if (NOT EXISTS ${CHAI_DIR})
    message(INFO ": Using chai from thirdPartyLibs")
    set(CHAI_DIR "${GEOSX_TPL_DIR}/chai" CACHE PATH "")
endif()

if (EXISTS ${CHAI_DIR})
    message(INFO ": Using chai found at ${CHAI_DIR}")
    
    find_library( CHAI_LIBRARY NAMES chai libchai libumpire libumpire_op libumpire_resource libumpire_strategy libumpire_tpl_judy libumpire_util
                  PATHS ${CHAI_DIR}/lib
                  NO_DEFAULT_PATH
                  NO_CMAKE_ENVIRONMENT_PATH
                  NO_CMAKE_PATH
                  NO_SYSTEM_ENVIRONMENT_PATH
                  NO_CMAKE_SYSTEM_PATH)

    include(${CHAI_DIR}/share/umpire/cmake/umpire-targets.cmake)
    include(${CHAI_DIR}/share/chai/cmake/chai-targets.cmake)

    # # handle the QUIETLY and REQUIRED arguments and set CHAI_FOUND to TRUE
    # # if all listed variables are TRUE
    set(CHAI_INCLUDE_DIRS ${CHAI_DIR}/include)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args( CHAI DEFAULT_MSG
                                       CHAI_INCLUDE_DIRS
                                       CHAI_LIBRARY )

    if (NOT CHAI_FOUND)
        message(FATAL_ERROR ": CHAI not found in ${CHAI_DIR}. Maybe you need to build it")
    endif()

    blt_register_library( NAME chai
                          INCLUDES ${CHAI_INCLUDE_DIRS}
                          LIBRARIES ${CHAI_LIBRARY}
                          TREAT_INCLUDES_AS_SYSTEM ON )

    set(ENABLE_CHAI ON CACHE BOOL "")
    set(thirdPartyLibs ${thirdPartyLibs} chai)
else()
    set(CHAI_FOUND FALSE)
    set(ENABLE_CHAI OFF CACHE BOOL "")
    message(INFO ": Not using chai")
endif()

set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )
