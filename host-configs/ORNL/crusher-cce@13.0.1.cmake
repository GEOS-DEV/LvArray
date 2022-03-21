set(CONFIG_NAME "crusher-cce@13.0.1" CACHE PATH "") 

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR "/gpfs/alpine/geo127/world-shared/cray-sles15-zen2/cce-13.0.1" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR} CACHE PATH "")

set(BLT_DIR "${GEOSX_TPL_DIR}/blt-0.4.1-3zz2mkf2wvglevvl4ozepe4tzhwtchoa/" CACHE PATH "" )
set(CAMP_DIR "${GEOSX_TPL_DIR}/camp-0.2.2-3qpdz6h2dzvfm5t7uabpz2ykiheza5b4/" CACHE PATH "" )

set(ENABLE_UMPIRE TRUE CACHE BOOL "" )
set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire-6.0.0-aeczo5gctizktwwt5x7xlmuyoarwipag/" CACHE PATH "" )
set(RAJA_DIR "${GEOSX_TPL_DIR}/raja-0.14.0-twro7k3cfsmp7s6mkiugsqncivj6w327/" CACHE PATH "" )
set(ENABLE_CHAI TRUE CACHE BOOL "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai-2.4.0-yubforuougga3ujwwpfz3tmybqhroczp/" CACHE PATH "" )

set(METIS_DIR "${GEOSX_TPL_DIR}/metis-5.1.0-zcfkawg5ifqpzcihrc3i6cdrrijusc2p/" CACHE PATH "" )
set(PARMETIS_DIR "${GEOSX_TPL_DIR}/parmetis-4.0.3-t2amifl5hh7yewre24gn2x3mlrz7qkl5/" CACHE PATH "" )

# C++ options
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.13/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/2.7.13/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/2.7.13/bin/ftn" CACHE PATH "")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

set( ENABLE_MPI ON CACHE BOOL "" FORCE )
set( ENABLE_FIND_MPI ON CACHE BOOL "" FORCE )

# HIP Options
set( ENABLE_HIP ON CACHE BOOL "" FORCE )
set( HIP_ROOT "/opt/rocm-4.5.2" CACHE PATH "" )
set( HIP_VERSION_STRING "4.5.2" CACHE STRING "" )

set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
set( AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE )

#set( CMAKE_CXX_FLAGS "--offload-arch=gfx90a -x hip -D__HIP_ROCclr -D__HIP_ARCH_GFX90A__=1" CACHE STRING "" FORCE )

set( HIP_HIPCC_INCLUDE_ARGS "$<$<BOOL:/opt/cray/pe/mpich/8.1.12/ofi/crayclang/10.0/include>:-I/opt/cray/pe/mpich/8.1.12/ofi/crayclang/10.0/include>" CACHE STRING "" FORCE )
set( HIP_HIPCC_FLAGS "-std=c++14 --amdgpu-target=gfx90a" CACHE STRING "" FORCE ) # -fgpu-rdc" CACHE STRING "" FORCE )

set( CMAKE_EXE_LINKER_FLAGS "-L/opt/cray/pe/mpich/8.1.12/ofi/crayclang/10.0/lib -lmpi -L/opt/cray/pe/mpich/8.1.12/gtl/lib -lmpi_gtl_hsa" CACHE STRING "" FORCE ) # -fpgu-rdc --hip-link
set( CMAKE_CXX_LINK_FLAGS ${CMAKE_EXE_LINKER_FLAGS} )

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")

set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
#set(DISABLE_UNIT_TESTS ON CACHE BOOL "" FORCE)
set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)

#BLT
set(ENABLE_FIND_MPI FALSE CACHE BOOL "")