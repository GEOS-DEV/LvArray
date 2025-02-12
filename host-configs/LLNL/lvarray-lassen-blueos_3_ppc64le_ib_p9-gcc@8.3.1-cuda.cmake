#################################################################################
# Generated host-config - Edit at own risk!
#################################################################################
#--------------------------------------------------------------------------------
# SYS_TYPE: blueos_3_ppc64le_ib_p9
# Compiler Spec: gcc@=8.3.1
# CMake executable path: /usr/tce/packages/cmake/cmake-3.29.2/bin/cmake
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Compilers
#--------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/g++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "")

#--------------------------------------------------------------------------------
# Cuda
#--------------------------------------------------------------------------------

set(ENABLE_CUDA ON CACHE BOOL "")

set(CMAKE_CUDA_STANDARD "17" CACHE PATH "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.8.0" CACHE PATH "")

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations -arch sm_70" CACHE STRING "")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -mcpu=powerpc64le -Xcompiler -mtune=powerpc64le" CACHE STRING "")

set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

#--------------------------------------------------------------------------------
# Performance Portability TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CHAI ON CACHE BOOL "")

set(CHAI_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/lassen-gcc-8-cuda-11_tpls/gcc-8.3.1/chai-git.df7741f1dbbdc5fff5f7d626151fdf1904e62b19_develop-jvnmbz3nfrjym77m2m2ddxfgyigf2hqv" CACHE PATH "")

set(RAJA_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/lassen-gcc-8-cuda-11_tpls/gcc-8.3.1/raja-git.4d7fcba55ebc7cb972b7cc9f6778b48e43792ea1_develop-ktim3j27la76tmlr6ryj2vjfllltmz4o" CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")

set(UMPIRE_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/lassen-gcc-8-cuda-11_tpls/gcc-8.3.1/umpire-git.abd729f40064175e999a83d11d6b073dac4c01d2_develop-xmi5nbcsff7dz76ljlmr5catjxa4hzjn" CACHE PATH "")

set(CAMP_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/lassen-gcc-8-cuda-11_tpls/gcc-8.3.1/camp-git.0f07de4240c42e0b38a8d872a20440cb4b33d9f5_main-d4ymmapzrvlzw6jwjiyenqd5l2sy7pg7" CACHE PATH "")

#--------------------------------------------------------------------------------
# IO TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CALIPER ON CACHE BOOL "")

set(CALIPER_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/lassen-gcc-8-cuda-11_tpls/gcc-8.3.1/caliper-git.287b7f3ad2d12f520aad04268d44f353cd05403c_2.12.0-rw6xo4fv4xn4oapuiypwwn7cz3m73naz" CACHE PATH "")

set(adiak_DIR "/usr/WS1/GEOS/GEOSX/TPLs_2025-02-12/lassen-gcc-8-cuda-11_tpls/gcc-8.3.1/adiak-0.4.0-n3yugcwz5o3ggvi2nrwybgmfattinedh/lib/cmake/adiak" CACHE PATH "")

#--------------------------------------------------------------------------------
# Documentation
#--------------------------------------------------------------------------------

set(ENABLE_DOXYGEN OFF CACHE BOOL "")

set(ENABLE_SPHINX OFF CACHE BOOL "")

#--------------------------------------------------------------------------------
# Development tools
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# addr2line
#--------------------------------------------------------------------------------

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

