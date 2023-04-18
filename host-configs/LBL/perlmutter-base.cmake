# Set up the tpls
set( GEOSX_TPL_ROOT_DIR /global/homes/d/dudouit/Projects/thirdPartyLibs CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR}/install-${CONFIG_NAME}-release CACHE PATH "")

set(CAMP_DIR ${GEOSX_TPL_DIR}/raja CACHE PATH "")
set(RAJA_DIR ${GEOSX_TPL_DIR}/raja CACHE PATH "")
set( RAJA_ENABLE_VECTORIZATION OFF CACHE BOOL "" FORCE)

set(ENABLE_UMPIRE ON CACHE BOOL "")
set(UMPIRE_DIR ${GEOSX_TPL_DIR}/chai CACHE PATH "")

set(ENABLE_CHAI ON CACHE BOOL "")
set(CHAI_DIR ${GEOSX_TPL_DIR}/chai CACHE PATH "")

set(ENABLE_CALIPER ON CACHE BOOL "")
set(CALIPER_DIR ${GEOSX_TPL_DIR}/caliper CACHE PATH "")

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
#set(CUDA_TOOLKIT_ROOT_DIR /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7 CACHE STRING "")
#set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
#set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "")
set(CUDA_ARCH sm_80 CACHE STRING "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

set(CHAI_CUDA_FLAGS "-arch ${CUDA_ARCH}" CACHE STRING "" FORCE)

# Uncomment this line to make nvcc output register usage for each kernel.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")

# Documentation
set(ENABLE_UNCRUSTIFY OFF CACHE BOOL "" FORCE)
set(ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
