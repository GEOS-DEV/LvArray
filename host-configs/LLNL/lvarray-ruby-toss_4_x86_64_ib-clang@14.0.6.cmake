#################################################################################
# Generated host-config - Edit at own risk!
#################################################################################
#--------------------------------------------------------------------------------
# SYS_TYPE: toss_4_x86_64_ib
# Compiler Spec: clang@=14.0.6
# CMake executable path: /usr/tce/backend/installations/linux-rhel8-x86_64/gcc-10.3.1/cmake-3.26.3-nz532rvfpaf5lf74zxmplgiobuhol7lu/bin/cmake
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Compilers
#--------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-14.0.6-magic/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-14.0.6-magic/bin/clang++" CACHE PATH "")

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

set(CHAI_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/chai-git.df7741f1dbbdc5fff5f7d626151fdf1904e62b19_develop-thhf2i6gjy3pnvdpgxf63zm2hulvdbwu" CACHE PATH "")

set(RAJA_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/raja-git.4d7fcba55ebc7cb972b7cc9f6778b48e43792ea1_develop-7bkwfqjvbwyfg6kwi3qnu5nbwoyu7c5s" CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")

set(UMPIRE_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/umpire-git.abd729f40064175e999a83d11d6b073dac4c01d2_develop-qc6ke6ra4bw5ieyx2eiitfkzngns6b2n" CACHE PATH "")

set(CAMP_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/camp-git.0f07de4240c42e0b38a8d872a20440cb4b33d9f5_main-4o6q4kk5tjy3tmlcgpns2w2ss5ibm5x6" CACHE PATH "")

#--------------------------------------------------------------------------------
# IO TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CALIPER ON CACHE BOOL "")

set(CALIPER_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/caliper-git.287b7f3ad2d12f520aad04268d44f353cd05403c_2.12.0-d6jzlhx4ofqairetauzhiw7rcocb3lj5" CACHE PATH "")

set(adiak_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/adiak-0.4.0-vxgcukydyqrm6ciuxrw4leflputw6r5e/lib/cmake/adiak" CACHE PATH "")

#--------------------------------------------------------------------------------
# Documentation
#--------------------------------------------------------------------------------

set(SPHINX_EXECUTABLE "/usr/gapps/GEOSX/thirdPartyLibs/python/quartz-gcc-python/python/bin/sphinx-build" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/ruby-clang-14_tpls/clang-14.0.6/doxygen-1.8.20-xj3wmf2ztwfreg3ao4zdjyjlami7lrb4/bin/doxygen" CACHE PATH "")

#--------------------------------------------------------------------------------
# Development tools
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# addr2line
#--------------------------------------------------------------------------------

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

