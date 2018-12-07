
set( thirdPartyLibs "")


################################
# AXOM
################################
if( EXISTS ${AXOM_DIR} AND ${ENABLE_AXOM} )

    MESSAGE( "Using AXOM_DIR=${AXOM_DIR}")
    set(AXOM_CMAKE ${AXOM_DIR}/lib/cmake)


    if(NOT EXISTS ${AXOM_CMAKE}/core-targets.cmake)
        MESSAGE(FATAL_ERROR "Could not find AXOM cmake include file (${AXOM_CMAKE}/core-targets.cmake)")
    endif()
    include(${AXOM_CMAKE}/core-targets.cmake)


    #if(ENABLE_MPI)
      if(NOT EXISTS ${AXOM_CMAKE}/lumberjack-targets.cmake)
          MESSAGE(FATAL_ERROR "Could not find AXOM cmake include file (${AXOM_CMAKE}/lumberjack-targets.cmake)")
      endif()
    include(${AXOM_CMAKE}/lumberjack-targets.cmake)
    #endif(ENABLE_MPI)

    if(NOT EXISTS ${AXOM_CMAKE}/slic-targets.cmake)
        MESSAGE(FATAL_ERROR "Could not find AXOM cmake include file (${AXOM_CMAKE}/slic-targets.cmake)")
    endif()
    include(${AXOM_CMAKE}/slic-targets.cmake)

    set(AXOM_FOUND TRUE)    
    
    set(AXOM_INCLUDE_DIRS ${AXOM_DIR}/include)
    
    
    blt_register_library( NAME slic
                          INCLUDES ${AXOM_INCLUDE_DIRS} 
                          LIBRARIES  slic
                          TREAT_INCLUDES_AS_SYSTEM ON)

    blt_register_library( NAME lumberjack
                          INCLUDES ${AXOM_INCLUDE_DIRS} 
                          LIBRARIES  lumberjack
                          TREAT_INCLUDES_AS_SYSTEM ON)
                        
    set( thirdPartyLibs ${thirdPartyLibs} sidre slic )
    
else()
    set(AXOM_FOUND FALSE)
endif()



################################
# CHAI
################################
if( ${ENABLE_CHAI})
  if( EXISTS ${CHAI_DIR})
      message("Using system CHAI found at ${CHAI_DIR}")
      set(CHAI_FOUND TRUE)
  endif()

#  find_path( CHAI_INCLUDE_DIRS chai/ManagedArray.hpp
   find_path( CHAI_INCLUDE_DIRS ${CHAI_DIR}/include
             PATHS  ${CHAI_DIR}/include
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

  find_library( CHAI_LIBRARY NAMES chai libchai libumpire libumpire_op libumpire_resource libumpire_strategy libumpire_tpl_judy libumpire_util
                PATHS ${CHAI_DIR}/lib
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH)


  include(FindPackageHandleStandardArgs)

  include("${CHAI_DIR}/share/umpire/cmake/umpire-targets.cmake")
  include("${CHAI_DIR}/share/chai/cmake/chai-targets.cmake")

  # handle the QUIETLY and REQUIRED arguments and set CHAI_FOUND to TRUE
  # if all listed variables are TRUE
  set(CHAI_FOUND TRUE)

  find_package_handle_standard_args( CHAI  DEFAULT_MSG
                                     CHAI_INCLUDE_DIRS
                                     CHAI_LIBRARY )


  if ( NOT CHAI_FOUND)
      message(FATAL_ERROR ": CHAI not found in ${CHAI_DIR}. Maybe you need to build it")
  endif()    

  blt_register_library( NAME chai
                        INCLUDES ${CHAI_INCLUDE_DIRS}
                        LIBRARIES ${CHAI_LIBRARY}
                        TREAT_INCLUDES_AS_SYSTEM ON )

  set( thirdPartyLibs ${thirdPartyLibs} chai )
else()
  message("Not using CHAI")
endif()






set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )