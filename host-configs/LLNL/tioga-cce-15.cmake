
set(CONFIG_NAME "tioga-cce@15.0.0c" CACHE PATH "")
include( ${CMAKE_CURRENT_LIST_DIR}/tioga-base.cmake )

# Set up the tpls
set(TPL_INSTALL_DATE 2023-05-17)
set(GEOSX_TPL_DIR "/usr/WS1/GEOS/GEOSX/TPLs_${TPL_INSTALL_DATE}/install-${CONFIG_NAME}" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR}/camp-main" CACHE PATH "" )
set(RAJA_DIR "${GEOSX_TPL_DIR}/raja-develop" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire-develop" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai-develop" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis-5.1.0" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis-4.0.3" CACHE PATH "" )

# C++ options
set(CRAYPE_VERSION "2.7.19")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

if( ENABLE_HIP )
  set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE )

  set( HIP_VERSION_STRING "5.4.0" CACHE STRING "" )
  set( HIP_ROOT "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
  set( ROCM_PATH ${HIP_ROOT} CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-munsafe-fp-atomics -fgpu-rdc" CACHE STRING "" FORCE )
  set( CMAKE_CXX_LINK_FLAGS "-fgpu-rdc --hip-link" CACHE STRING "" FORCE )
endif()
