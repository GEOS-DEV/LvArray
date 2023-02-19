include( ${CMAKE_CURRENT_LIST_DIR}/crusher-base.cmake )

set(CCE_VERSION 15.0.0)
set(CONFIG_NAME "crusher-cce@${CCE_VERSION}" CACHE PATH "")

# Set up the tpls
set(GEOSX_TPL_DIR "/gpfs/alpine/geo127/world-shared/spack/opt/2023-02-17/linux-sles15-zen3/cce-${CCE_VERSION}" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR}/camp" CACHE PATH "" )
set(RAJA_DIR "${GEOSX_TPL_DIR}/raja" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis" CACHE PATH "" )

# C++ options
set(CRAYPE_VERSION "2.7.16")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

if( ENABLE_HIP )
  set( ROCM_VERSION 5.4.0 CACHE STRING "" )
  set( HIP_ROOT "/opt/rocm-${ROCM_VERSION}" CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-munsafe-fp-atomics -fno-gpu-rdc" CACHE STRING "" FORCE ) #-mno-unsafe-fp-atomics
  set( CMAKE_CXX_LINK_FLAGS "-munsafe-fp-atomics -fno-gpu-rdc --hip-link" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS_DEBUG "-O1 -g ${CMAKE_CXX_FLAGS}" CACHE STRING "" )

endif()
