include( ${CMAKE_CURRENT_LIST_DIR}/frontier-base.cmake )

set(CCE_VERSION 15.0.0)
set(CONFIG_NAME "frontier-cce@${CCE_VERSION}" CACHE PATH "")

# Set up the tpls
set( GEOSX_TPL_DATE 2023-04-04 )
set( GEOSX_TPL_DIR "/lustre/orion/geo127/world-shared/tpls/${GEOSX_TPL_DATE}/install-${CONFIG_NAME}/" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR}/camp-2022.03.2" CACHE PATH "" )
set(RAJA_DIR "${GEOSX_TPL_DIR}/raja-2022.03.0" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire-2022.03.0" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai-2022.03.0" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis-5.1.0" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis-4.0.3" CACHE PATH "" )

# C++ options
set(CRAYPE_VERSION "2.7.16")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

if( ENABLE_HIP )
  set( ROCM_VERSION 5.4.3 CACHE STRING "" )
  set( HIP_ROOT "/opt/rocm-${ROCM_VERSION}" CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-munsafe-fp-atomics -fno-gpu-rdc" CACHE STRING "" FORCE ) #-mno-unsafe-fp-atomics
  set( CMAKE_CXX_LINK_FLAGS "-munsafe-fp-atomics -fno-gpu-rdc --hip-link" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS_DEBUG "-O1 -g ${CMAKE_CXX_FLAGS}" CACHE STRING "" )

endif()
