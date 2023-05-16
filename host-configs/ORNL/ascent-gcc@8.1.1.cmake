set(CONFIG_NAME "ascent-gcc@8.1.1" CACHE PATH "") 

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR "/ccsopen/proj/gen136/GEOSX/thirdPartyLibs" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR}/install-${CONFIG_NAME}-release CACHE PATH "")

# C++ options
set(CMAKE_CXX_COMPILER "/sw/ascent/gcc/8.1.1/bin/g++" CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le -Wno-strict-aliasing" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wno-strict-aliasing" CACHE STRING "")
set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

# OpenMP options
#set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)
#set(OpenMP_Fortran_FLAGS "-fopenmp" CACHE STRING "")
#set(OpenMP_Fortran_LIB_NAMES "" CACHE STRING "")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR "/sw/ascent/cuda/10.1.243" CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER} CACHE STRING "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE STRING "")
set(CUDA_ARCH sm_70 CACHE STRING "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
# -std=c++14 needed to work around cuda/gcc bug
set(CMAKE_CUDA_FLAGS "-std=c++14 -Xcompiler -mno-float128 -restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -mcpu=powerpc64le -Xcompiler -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")
 
# Uncomment this line to make nvcc output register usage for each kernel.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")
