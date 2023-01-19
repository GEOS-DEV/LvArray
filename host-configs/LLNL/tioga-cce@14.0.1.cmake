
set(CONFIG_NAME "tioga-cce@14.0.1" CACHE PATH "")
include( ${CMAKE_CURRENT_LIST_DIR}/tioga-base.cmake )

# Set up the tpls
set(TPL_INSTALL_DATE 2023-01-19)
set(GEOSX_TPL_DIR "/usr/WS1/GEOS/GEOSX/TPLs_${TPL_INSTALL_DATE}/install-${CONFIG_NAME}" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR}/camp" CACHE PATH "" )
set(RAJA_DIR "${GEOSX_TPL_DIR}/raja" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis" CACHE PATH "" )

# C++ options
set(CRAYPE_VERSION "2.7.19")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

if( ENABLE_HIP )
  set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE ) # don't invoke hipcc, rely on cce link-time compilation

  set( HIP_VERSION_STRING "5.1.0" CACHE STRING "" )
  set( HIP_ROOT "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
  set( hip_DIR "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-mno-unsafe-fp-atomics -fgpu-rdc" CACHE STRING "" FORCE )
  set( CMAKE_CXX_LINK_FLAGS "-fgpu-rdc --hip-link" CACHE STRING "" FORCE )
endif()
