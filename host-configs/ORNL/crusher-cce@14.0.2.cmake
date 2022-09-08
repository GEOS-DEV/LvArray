
set(CONFIG_NAME "crusher-cce@14.0.2" CACHE PATH "")
include( ${CMAKE_CURRENT_LIST_DIR}/crusher-base.cmake )

# Set up the tpls
# set(SYSTEM_TPL_DIR "/sw/crusher/spack-envs/base/opt/cray-sles15-zen3/cce-14.0.2" CACHE PATH "")
set(GEOSX_TPL_DIR "/gpfs/alpine/geo127/world-shared/cray-sles15-zen3/cce-14.0.2" CACHE PATH "")

set(CAMP_DIR "${GEOSX_TPL_DIR}/camp-0.2.2-3why4qpjkpe2cdn5paq2kp5nyl34lxmo" CACHE PATH "" )

set(RAJA_DIR "${GEOSX_TPL_DIR}/raja-2022.03.0-d62zou3ifamctehpl4bdsmlzaxg2kkgk" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire-2022.03.1-ouoie6dc5kutik3qsafdan4hr5foo5hk" CACHE PATH "" )

set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai-2022.03.0-ebulxhddabg37ttym3g6iijbpz4z3zwz" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis-5.1.0-jptrwzs7vdbckndjg5qg4jwckfmgexmw/" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis-4.0.3-p2msdgsmomufcnwhnow5bbazg7463caf/" CACHE PATH "" )

# C++ options
set(CRAYPE_VERSION "2.7.17")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

if( ENABLE_HIP )
  set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE ) # don't invoke hipcc, rely on cce link-time compilation

  set( HIP_VERSION_STRING "5.1.0" CACHE STRING "" )
  set( HIP_ROOT "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )

  set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
  set( AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE )
  set( CMAKE_CXX_FLAGS "-fgpu-rdc" CACHE STRING "" FORCE )
  set( CMAKE_CXX_LINK_FLAGS "-fgpu-rdc --hip-link" CACHE STRING "" FORCE )
endif()
