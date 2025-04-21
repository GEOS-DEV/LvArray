#################################################################################
# Generated host-config - Edit at own risk!
#################################################################################
#--------------------------------------------------------------------------------
# SYS_TYPE: toss_4_x86_64_ib
# Compiler Spec: gcc@=12noAVX
# CMake executable path: /usr/tce/backend/installations/linux-rhel8-x86_64/gcc-10.3.1/cmake-3.26.3-nz532rvfpaf5lf74zxmplgiobuhol7lu/bin/cmake
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Compilers
#--------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-12.1.1-magic/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-12.1.1-magic/bin/g++" CACHE PATH "")

set(CMAKE_CXX_FLAGS "-march=x86-64-v2 -mno-avx512f" CACHE PATH "")

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

set(CHAI_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/chai-git.4b9060b18b9bec1167026cfb3132bd540c4bd56b_develop-leifvg5jj3pqsdbt2jvujdotbvjavidq" CACHE PATH "")

set(RAJA_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/raja-git.1d70abf171474d331f1409908bdf1b1c3fe19222_develop-2y47wpygfea35spibzd657rncwjgr3xk" CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")

set(UMPIRE_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/umpire-git.1ed0669c57f041baa1f1070693991c3a7a43e7ee_develop-2fsgks5hmrb5eep4xrcn7cn6tf2s25qt" CACHE PATH "")

set(CAMP_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/camp-git.ee0a3069a7ae72da8bcea63c06260fad34901d43_main-sw3zpyoowyj5f2eo43rnd5sxmdeopmzi" CACHE PATH "")

#--------------------------------------------------------------------------------
# IO TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CALIPER ON CACHE BOOL "")

set(CALIPER_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/caliper-git.287b7f3ad2d12f520aad04268d44f353cd05403c_2.12.0-olusdj2xpppf5tam6ktcysn6ogeq4i2d" CACHE PATH "")

set(adiak_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/adiak-0.4.0-kgrknuu6drw34t7svfcen5t5u37i4jnf/lib/cmake/adiak" CACHE PATH "")

#--------------------------------------------------------------------------------
# Documentation
#--------------------------------------------------------------------------------

set(SPHINX_EXECUTABLE "/usr/gapps/GEOSX/thirdPartyLibs/python/quartz-gcc-python/python/bin/sphinx-build" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/ruby-gcc-12noAVX_tpls/gcc-12noAVX/doxygen-1.8.20-c4zarmc366msdoizvmau2bs7n76ob7vo/bin/doxygen" CACHE PATH "")

#--------------------------------------------------------------------------------
# Development tools
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# addr2line
#--------------------------------------------------------------------------------

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

