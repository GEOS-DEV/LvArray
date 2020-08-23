set(CONFIG_NAME "lassen-xl@2020.08.13" CACHE PATH "")

# set(RAJA_DIR PATH/TO/RAJA CACHE PATH "")

set(ENABLE_UMPIRE OFF CACHE BOOL "")
set(ENABLE_CHAI OFF CACHE BOOL "")
set(ENABLE_CALIPER OFF CACHE BOOL "")
set(USE_ADDR2LINE ON CACHE BOOL "")

# C options
set(CMAKE_C_COMPILER /usr/tce/packages/xl/xl-2020.08.13-cuda-11.0.2/bin/xlc CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -qxlcompatmacros -qlanglvl=extended0x -qalias=noansi -qsmp=omp -qhot -qnoeh -qsuppress=1500-029 -qsuppress=1500-036" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -qxlcompatmacros -qlanglvl=extended0x -qalias=noansi -qsmp=omp -qhot -qnoeh -qsuppress=1500-029 -qsuppress=1500-036" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

# C++ options
set(CMAKE_CXX_COMPILER /usr/tce/packages/xl/xl-2020.08.13-cuda-11.0.2/bin/xlC CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -qxlcompatmacros -qlanglvl=extended0x -qalias=noansi -qsmp=omp -qhot -qnoeh -qsuppress=1500-029 -qsuppress=1500-036" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -qxlcompatmacros -qlanglvl=extended0x -qalias=noansi -qsmp=omp -qhot -qnoeh -qsuppress=1500-029 -qsuppress=1500-036" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(BLT_CXX_STANDARD 14 CACHE STRING "")

# OpenMP options
set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)

# MPI options
set(ENABLE_MPI ON CACHE BOOL "")
set(MPI_ROOT /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-2020.08.13-cuda-11.0.2/ CACHE PATH "")
set(MPI_C_COMPILER         ${MPI_ROOT}/bin/mpicc  CACHE PATH "")
set(MPI_CXX_COMPILER       ${MPI_ROOT}/bin/mpicxx CACHE PATH "")
set(MPIEXEC                lrun CACHE STRING "")
set(MPIEXEC_NUMPROC_FLAG   -n CACHE STRING "")
set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC ON CACHE BOOL "")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR /usr/tce/packages/cuda/cuda-11.0.2 CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER} CACHE STRING "")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set(CUDA_ARCH sm_70 CACHE STRING "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -qxlcompatmacros -Xcompiler -qalias=noansi -Xcompiler -qsmp=omp -Xcompiler -qhot -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-036" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

# Uncomment this line to make nvcc output register usage for each kernel.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")

# Documentation options
set(ENABLE_DOCS OFF CACHE BOOL "")
