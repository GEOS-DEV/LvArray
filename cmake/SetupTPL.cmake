macro(find_and_register)
    set(singleValueArgs NAME HEADER)
    set(multiValueArgs INCLUDE_DIRECTORIES
                       LIBRARY_DIRECTORIES
                       LIBRARIES
                       EXTRA_LIBRARIES
                       DEPENDS )

    ## parse the arguments
    cmake_parse_arguments(arg
                          "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_NAME)
        message(FATAL_ERROR "The find_and_register required parameter NAME specifies the name of the library to register.")
    endif()

    if(NOT DEFINED arg_INCLUDE_DIRECTORIES)
        message(FATAL_ERROR "The find_and_register required parameter INCLUDE_DIRECTORIES specifies the directories to search for the given header.")
    endif()

    if(NOT DEFINED arg_LIBRARY_DIRECTORIES)
        message(FATAL_ERROR "The find_and_register required parameter LIBRARY_DIRECTORIES specifies the directories to search for the given libraries.")
    endif()

    if(NOT DEFINED arg_HEADER)
        message(FATAL_ERROR "The find_and_register required parameter HEADER specifies the header to search for.")
    endif()

    if(NOT DEFINED arg_LIBRARIES)
        message(FATAL_ERROR "The find_and_register required parameter LIBRARIES specifies the libraries to search for.")
    endif()

    find_path(${arg_NAME}_INCLUDE_DIR ${arg_HEADER}
              PATHS ${arg_INCLUDE_DIRECTORIES}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

    if(${arg_NAME}_INCLUDE_DIR STREQUAL ${arg_NAME}_INCLUDE_DIR-NOTFOUND)
        message(FATAL_ERROR "Could not find '${arg_HEADER}' in '${arg_INCLUDE_DIRECTORIES}'")
    endif()

    blt_find_libraries(FOUND_LIBS ${arg_NAME}_LIBRARIES
                       NAMES ${arg_LIBRARIES}
                       PATHS ${arg_LIBRARY_DIRECTORIES}
                       REQUIRED ON)

    blt_import_library(NAME ${arg_NAME}
                         INCLUDES ${${arg_NAME}_INCLUDE_DIR}
                         LIBRARIES ${${arg_NAME}_LIBRARIES} ${arg_EXTRA_LIBRARIES}
                         TREAT_INCLUDES_AS_SYSTEM ON
                         DEPENDS_ON ${arg_DEPENDS})

endmacro(find_and_register)

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
    if(NOT EXISTS ${CHAI_DIR})
        message(FATAL_ERROR "CHAI_DIR must be defined and point to a valid directory when using CHAI.")
    endif()

    message(STATUS "Using CHAI from ${CHAI_DIR}")

    if(NOT ENABLE_UMPIRE)
        message(FATAL_ERROR "Umpire must be enabled to use CHAI.")
    endif()

    if(NOT ENABLE_RAJA)
        message(FATAL_ERROR "RAJA must be enabled to use CHAI.")
    endif()

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
                         LIBRARIES caliper ${CUPTI_LIB}
                         TREAT_INCLUDES_AS_SYSTEM ON)

    set(thirdPartyLibs ${thirdPartyLibs} caliper)
else()
    message(STATUS "Not using caliper.")
endif()

################################
# Python
################################
if(ENABLE_PYLVARRAY)
    message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
    find_package(Python3 REQUIRED
                 COMPONENTS Development NumPy)

    message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_LIBRARY_DIRS = ${Python3_LIBRARY_DIRS}")
    message(STATUS "Python3_NumPy_INCLUDE_DIRS = ${Python3_NumPy_INCLUDE_DIRS}")

    set(thirdPartyLibs ${thirdPartyLibs} Python3::Python Python3::NumPy)
else()
    message(STATUS "Not building pylvarray")
endif()

################################
# LAPACK/BLAS
################################
if(ENABLE_LAPACK)
    message(STATUS "BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")

    blt_import_library(NAME blas
                       TREAT_INCLUDES_AS_SYSTEM ON
                       LIBRARIES ${BLAS_LIBRARIES})

    blt_import_library(NAME lapack
                       DEPENDS_ON blas
                       TREAT_INCLUDES_AS_SYSTEM ON
                       LIBRARIES ${LAPACK_LIBRARIES})

    set(thirdPartyLibs ${thirdPartyLibs} blas lapack)
else()
    message(STATUS "Not using LAPACK or BLAS.")
endif()

################################
# MAGMA
################################
if(ENABLE_MAGMA)
    message(STATUS "Using MAGMA from ${MAGMA_DIR}")

    if(NOT ENABLE_LAPACK)
        message(FATAL_ERROR "LAPACK must be enabled to use MAGMA.")
    endif()

    find_and_register(NAME magma
                      INCLUDE_DIRECTORIES ${MAGMA_DIR}/include
                      LIBRARY_DIRECTORIES ${MAGMA_DIR}/lib
                      HEADER magma.h
                      LIBRARIES magma)
    
    set(thirdPartyLibs ${thirdPartyLibs} magma)
else()
    message(STATUS "Not using MAGMA.")
endif()

set( thirdPartyLibs ${thirdPartyLibs} CACHE STRING "" )
