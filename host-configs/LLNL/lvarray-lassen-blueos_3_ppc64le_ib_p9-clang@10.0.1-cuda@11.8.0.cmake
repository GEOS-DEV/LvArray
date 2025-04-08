#################################################################################
# Generated host-config - Edit at own risk!
#################################################################################
#--------------------------------------------------------------------------------
# SYS_TYPE: blueos_3_ppc64le_ib_p9
# Compiler Spec: clang@=10.0.1
# CMake executable path: /usr/tce/packages/cmake/cmake-3.29.2/bin/cmake
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Compilers
#--------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/bin/clang++" CACHE PATH "")

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

set(CHAI_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/lassen-clang-10-cuda-11_tpls/clang-10.0.1/chai-git.4b9060b18b9bec1167026cfb3132bd540c4bd56b_develop-77slff2jnddra4feyt64f23fza2x6eeh" CACHE PATH "")

set(RAJA_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/lassen-clang-10-cuda-11_tpls/clang-10.0.1/raja-git.1d70abf171474d331f1409908bdf1b1c3fe19222_develop-4g4qjxcaas7stqgc5ilday63r5557jf2" CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")

set(UMPIRE_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/lassen-clang-10-cuda-11_tpls/clang-10.0.1/umpire-git.1ed0669c57f041baa1f1070693991c3a7a43e7ee_develop-oxerfmieb5y5mu6mutrl2syxmo3jxwsa" CACHE PATH "")

set(CAMP_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/lassen-clang-10-cuda-11_tpls/clang-10.0.1/camp-git.ee0a3069a7ae72da8bcea63c06260fad34901d43_main-qaxwkoxoc7xcuz6zdzjobll4mst7ia4a" CACHE PATH "")

#--------------------------------------------------------------------------------
# IO TPLs
#--------------------------------------------------------------------------------

set(ENABLE_CALIPER ON CACHE BOOL "")

set(CALIPER_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/lassen-clang-10-cuda-11_tpls/clang-10.0.1/caliper-git.287b7f3ad2d12f520aad04268d44f353cd05403c_2.12.0-xiwon23mamhn2tkcnqks5crznnqaovjt" CACHE PATH "")

set(adiak_DIR "/usr/gapps/GEOSX/thirdPartyLibs/2025-04-03_spack/lassen-clang-10-cuda-11_tpls/clang-10.0.1/adiak-0.4.0-tp3jxzzp7ifbreg2erbvusm34w64jau7/lib/cmake/adiak" CACHE PATH "")

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

