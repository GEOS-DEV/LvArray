
set(CONFIG_NAME "crusher-cce@13.0.1" CACHE PATH "")
include( ${CMAKE_CURRENT_LIST_DIR}/crusher-base.cmake )

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR "/gpfs/alpine/geo127/world-shared/cray-sles15-zen2/cce-13.0.1" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR} CACHE PATH "")
set(GEOSX_TPL_DIR2 "/gpfs/alpine/geo127/world-shared/cray-sles15-zen3/cce-13.0.1" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR2}/camp-0.2.2-oogry5gz2fts7jufeykxzmowajtmgzi3" CACHE PATH "" )

set(RAJA_DIR "${GEOSX_TPL_DIR2}/raja-2022.03.0-ex5v5y6jtotfxxvwcs7bblwvy4ktjykq" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR2}/umpire-develop-jqqth57w2ets75sljw7lc5uxoi5wwi3c" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR2}/chai-2022.03.0-w7lka3bkp36mbk5kzucgtp3eowomllgl" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis-5.1.0-zcfkawg5ifqpzcihrc3i6cdrrijusc2p/" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis-4.0.3-t2amifl5hh7yewre24gn2x3mlrz7qkl5/" CACHE PATH "" )

# C++ options
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.13/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/2.7.13/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/2.7.13/bin/ftn" CACHE PATH "")


if( ENABLE_HIP )
  set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE ) # don't invoke hipcc, rely on cce link-time compilation

  set( HIP_VERSION_STRING "4.5.2" CACHE STRING "" )
  set( HIP_ROOT "/opt/rocm-4.5.2" CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-fgpu-rdc" CACHE STRING "" FORCE )
  set( CMAKE_CXX_LINK_FLAGS "-fgpu-rdc --hip-link" CACHE STRING "" FORCE )
endif()
