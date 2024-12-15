set( CCE_VERSION_STRING "18.0.1" )
set( HIP_VERSION_STRING "6.2.1" )
set( CONFIG_NAME "tioga-cce-${CCE_VERSION_STRING}-rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
include( ${CMAKE_CURRENT_LIST_DIR}/tioga-base.cmake )

# TODO: Set up the tpls
#set(TPL_INSTALL_DATE 2023-05-17)
#set(GEOS_TPL_DIR "/usr/WS1/GEOS/GEOSX/TPLs_${TPL_INSTALL_DATE}/install-${CONFIG_NAME}" CACHE PATH "")

set( ENABLE_UMPIRE ON CACHE BOOL "" )
set( ENABLE_CHAI ON CACHE BOOL "" )
set( ENABLE_CALIPER ON CACHE BOOL "" )
set( ENABLE_ADIAK ON CACHE BOOL "" )

set( CAMP_DIR ${GEOS_TPL_DIR}/raja CACHE PATH "" )
set( RAJA_DIR ${GEOS_TPL_DIR}/raja CACHE PATH "" )
set( UMPIRE_DIR ${GEOS_TPL_DIR}/chai CACHE PATH "" )
set( CHAI_DIR ${GEOS_TPL_DIR}/chai CACHE PATH "" )
set( CALIPER_DIR ${GEOS_TPL_DIR}/caliper CACHE PATH "" )

# C++ options
set( CRAYPE_VERSION "2.7.33")
set( CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "" )
set( CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "" )
set( CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "" )

if( ENABLE_HIP )
  set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE )

  set( HIP_ROOT "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
  set( ROCM_PATH ${HIP_ROOT} CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-munsafe-fp-atomics -fgpu-rdc" CACHE STRING "" FORCE )
  set( CMAKE_CXX_LINK_FLAGS "-fgpu-rdc --hip-link" CACHE STRING "" FORCE )
endif()
