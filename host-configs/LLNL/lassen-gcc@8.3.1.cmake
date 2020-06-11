set(CONFIG_NAME "lassen-gcc@8.3.1" CACHE PATH "") 

# Set up the tpls
# These were probably built with clang (no guarantee that they would work)
set(GEOSX_TPL_ROOT_DIR /usr/gapps/GEOSX/thirdPartyLibs CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR}/2020-06-11/install-${CONFIG_NAME}-release CACHE PATH "")

set(CMAKE_C_COMPILER /usr/tce/packages/gcc/gcc-8.3.1/bin/gcc CACHE PATH "")
set(CMAKE_CXX_COMPILER /usr/tce/packages/gcc/gcc-8.3.1/bin/g++ CACHE PATH "")
set(CMAKE_Fortran_COMPILER /usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran CACHE PATH "")

# These flags should be set for the TPL build but they break some of our numerically sensitive code.
# This needs to be fixed.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=power9 -mtune=power9" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=power9 -mtune=power9" CACHE STRING "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=power9 -mtune=power9" CACHE STRING "")
set(FORTRAN_MANGLE_NO_UNDERSCORE OFF CACHE BOOL "")

# OpenMP options
set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)

# MPI options
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_ROOT /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1 CACHE PATH "")

set(MPI_C_COMPILER         ${MPI_ROOT}/bin/mpicc  CACHE PATH "")
set(MPI_CXX_COMPILER       ${MPI_ROOT}/bin/mpicxx CACHE PATH "")
set(MPI_Fortran_COMPILER   ${MPI_ROOT}/bin/mpifort CACHE PATH "")
set(MPIEXEC                lrun CACHE STRING "")
set(MPIEXEC_NUMPROC_FLAG   -n CACHE STRING "")
set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC ON CACHE BOOL "")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR /usr/tce/packages/cuda/cuda-10.1.243 CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER} CACHE STRING "")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set(CUDA_ARCH sm_70 CACHE STRING "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} -Xcompiler -mno-float128 --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -mcpu=powerpc64le -Xcompiler -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

# Uncomment this line to make nvcc output register usage for each kernel.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")
