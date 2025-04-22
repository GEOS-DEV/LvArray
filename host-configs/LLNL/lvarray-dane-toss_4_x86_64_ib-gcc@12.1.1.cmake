#################################################################################
# Generated host-config - Edit at own risk!
#################################################################################
#--------------------------------------------------------------------------------
# SYS_TYPE: toss_4_x86_64_ib
# Compiler Spec: gcc@=12.1.1
# CMake executable path: /usr/tce/backend/installations/linux-rhel8-x86_64/gcc-10.3.1/cmake-3.26.3-nz532rvfpaf5lf74zxmplgiobuhol7lu/bin/cmake
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Compilers
#--------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-12.1.1-magic/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-12.1.1-magic/bin/g++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "")

#--------------------------------------------------------------------------------
# Cuda
#--------------------------------------------------------------------------------

set(ENABLE_CUDA OFF CACHE BOOL "")

#--------------------------------------------------------------------------------
# Performance Portability TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CHAI ON CACHE BOOL "")

set(CHAI_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/chai-git.4b9060b18b9bec1167026cfb3132bd540c4bd56b_develop-uq6ogpe5shfs5s5ifayx27gvzm3n3xcf" CACHE PATH "")

set(RAJA_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/raja-git.1d70abf171474d331f1409908bdf1b1c3fe19222_develop-37yxlbkfjq4ww4iw46u3a3d52oi6pul4" CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")

set(UMPIRE_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/umpire-git.1ed0669c57f041baa1f1070693991c3a7a43e7ee_develop-tymd742qkyap4m2m3kcnlx4mqnikk4pw" CACHE PATH "")

set(CAMP_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/camp-git.ee0a3069a7ae72da8bcea63c06260fad34901d43_main-wtxnzy3oakj4hdtuj5vpwpdtymqmyt4s" CACHE PATH "")

#--------------------------------------------------------------------------------
# IO TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CALIPER ON CACHE BOOL "")

set(CALIPER_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/caliper-git.287b7f3ad2d12f520aad04268d44f353cd05403c_2.12.0-bno6d6qyyvivznkozzdh27ljrj3yqbv2" CACHE PATH "")

set(adiak_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/adiak-0.4.0-dimwte7ij4naho6hhcqowidlxnespn7e/lib/cmake/adiak" CACHE PATH "")

#--------------------------------------------------------------------------------
# Documentation
#--------------------------------------------------------------------------------

set(SPHINX_EXECUTABLE "/usr/gapps/GEOSX/thirdPartyLibs/python/quartz-gcc-python/python/bin/sphinx-build" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/dane-gcc-12_tpls/gcc-12.1.1/doxygen-1.8.20-hbxmvlkrwmpt5mvibhths6cdo5rlor3s/bin/doxygen" CACHE PATH "")

#--------------------------------------------------------------------------------
# Development tools
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# addr2line
#--------------------------------------------------------------------------------

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

