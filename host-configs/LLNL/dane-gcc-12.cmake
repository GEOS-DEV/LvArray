set(CONFIG_NAME "dane-gcc-12" CACHE PATH "")

# Set up the tpls
set( TPL_INSTALL_DATE 2024-12-06 )
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
set( GEOS_TPL_ROOT_DIR /usr/gapps/GEOSX/thirdPartyLibs CACHE PATH "" )
set( GEOS_TPL_DIR "${GEOS_TPL_ROOT_DIR}/${TPL_INSTALL_DATE}/install-${CONFIG_NAME}-${CMAKE_BUILD_TYPE_LOWER}" CACHE PATH "")

message( STATUS " " )
message( STATUS "GEOS_TPL_DIR: ${GEOS_TPL_DIR}" )
message( STATUS " " )

include(${CMAKE_CURRENT_LIST_DIR}/llnl-cpu-gcc-12.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dane-base.cmake)
