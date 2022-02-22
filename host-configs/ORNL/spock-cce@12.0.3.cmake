set(CONFIG_NAME "spock-cce@12.0.3" CACHE PATH "") 

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR "/gpfs/alpine/geo127/scratch/tobin6/spack/opt/spack/cray-sles15-zen2/cce-12.0.3" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR} CACHE PATH "")

set(BLT_DIR "${GEOSX_TPL_DIR}/blt-0.4.1-qpmhf6p7n5sarmks55hgjnzff3ncs7jd/" CACHE PATH "" )
set(CAMP_DIR "${GEOSX_TPL_DIR}/camp-0.2.2-frggdmwjevbxy4a6kw7ctgrhyv7erfhr/" CACHE PATH "" )

set(UMPIRE_DIR "${GEOSX_TPL_DIR}/umpire-6.0.0-nkdetdg5tjyzzf5yjzo32jxwkmwfjjqn/" CACHE PATH "" )
set(RAJA_DIR "${GEOSX_TPL_DIR}/raja-0.14.0-wun25mr5qf7vo6x2vblhzh2ivs7vr4g6/" CACHE PATH "" )
set(CHAI_DIR "${GEOSX_TPL_DIR}/chai-2.4.0-a5ponjo23u7smy7w4a4jj7im47shrsxk/" CACHE PATH "" )

set(METIS_DIR "/sw/spock/spack-envs/base/opt/cray-sles15-zen2/cce-12.0.3/metis-5.1.0-rbblqiymq6eoursordyaq2ghimzpd22v/" CACHE PATH "" )
set(PARMETIS_DIR "/sw/spock/spack-envs/base/opt/cray-sles15-zen2/cce-12.0.3/parmetis-4.0.3-mliemgo6vxrahsz4f6u5agdqyfpk2yd2/" CACHE PATH "" )

# C++ options
#set(CMAKE_C_COMPILER "/opt/cray/pe/cce/12.0.3/bin/craycc" CACHE PATH "")
#set(CMAKE_CXX_COMPILER "/opt/cray/pe/cce/12.0.3/bin/crayCC" CACHE PATH "")
#set(CMAKE_Fortran_COMPILER "/opt/cray/pe/cce/12.0.3/bin/crayftn" CACHE PATH "")

set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.11/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/2.7.11/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/2.7.11/bin/ftn" CACHE PATH "")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

set( ENABLE_MPI ON CACHE BOOL "" FORCE )
set( ENABLE_FIND_MPI OFF CACHE BOOL "" FORCE )

# HIP Options
set( ENABLE_HIP ON CACHE BOOL "" FORCE )
set( HIP_ROOT "/opt/rocm-4.2.0" CACHE PATH "" )
set( HIP_VERSION_STRING "4.2.0" CACHE STRING "" )
set( CMAKE_HIP_ARCHITECTURES "gfx908" CACHE STRING "" FORCE )

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")
