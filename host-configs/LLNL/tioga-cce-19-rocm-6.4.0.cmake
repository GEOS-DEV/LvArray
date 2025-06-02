set( CCE_VERSION_STRING "19.0.0" )
set( HIP_VERSION_STRING "6.4.0" )
set( CONFIG_NAME "tioga-cce-${CCE_VERSION_STRING}-rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
include( ${CMAKE_CURRENT_LIST_DIR}/amdgpu-base.cmake )

# TODO: Set up GEOS_TPL_DIR
set( ENABLE_UMPIRE ON CACHE BOOL "" )
set( ENABLE_CHAI ON CACHE BOOL "" )
set( ENABLE_CALIPER ON CACHE BOOL "" )
set( ENABLE_ADIAK ON CACHE BOOL "" )

set( CAMP_DIR ${GEOS_TPL_DIR}/raja CACHE PATH "" )
set( RAJA_DIR ${GEOS_TPL_DIR}/raja CACHE PATH "" )
set( UMPIRE_DIR ${GEOS_TPL_DIR}/chai CACHE PATH "" )
set( CHAI_DIR ${GEOS_TPL_DIR}/chai CACHE PATH "" )
set( CALIPER_DIR ${GEOS_TPL_DIR}/caliper CACHE PATH "" )

# MPI options
set( MPI_HOME /opt/cray/pe/mpich/8.1.33.1/ofi/crayclang/18.0 CACHE PATH "" )
set( MPI_INCLUDE_DIR ${MPI_HOME}/include CACHE PATH "" ) # Needed by hypre

# C++ options
set( CRAYPE_VERSION "2.7.34")
set( CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "" )
set( CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "" )
set( CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "" )

if( ENABLE_HIP )
  set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE )

  set( HIP_ROOT "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
  set( HIP_ROOT_DIR ${HIP_ROOT} CACHE PATH "" )
  set( ROCM_PATH ${HIP_ROOT} CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( CMAKE_HIP_FLAGS "-munsafe-fp-atomics -fno-gpu-rdc -I${MPI_INCLUDE_DIR}" CACHE STRING "" FORCE )
  set( CMAKE_HIP_LINK_FLAGS "-fno-gpu-rdc --hip-link -Wl,--allow-shlib-undefined" CACHE STRING "" FORCE )
endif()
