set(CONFIG_NAME "lassen-clang@upstream" CACHE PATH "") 

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR /usr/gapps/GEOSX/thirdPartyLibs CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR}/2020-09-18/install-${CONFIG_NAME}-release CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")
set(ENABLE_CHAI ON CACHE BOOL "")
set(USE_ADDR2LINE ON CACHE BOOL "")

# C options
set(CMAKE_C_COMPILER /usr/tce/packages/clang/clang-upstream-2019.03.26/bin/clang CACHE PATH "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g ${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

# C++ options
set(CMAKE_CXX_COMPILER /usr/tce/packages/clang/clang-upstream-2019.03.26/bin/clang++ CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")
set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

# Fortran options
set(CMAKE_Fortran_COMPILER /usr/tce/packages/xl/xl-beta-2019.06.20/bin/xlf_r CACHE PATH "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG -qarch=pwr9 -qtune=pwr9" CACHE STRING "")
set(FORTRAN_MANGLE_NO_UNDERSCORE ON CACHE BOOL "")

# OpenMP options
set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)
set(OpenMP_Fortran_FLAGS "-qsmp=omp" CACHE STRING "")
set(OpenMP_Fortran_LIB_NAMES "" CACHE STRING "")

# MPI options
set(ENABLE_MPI ON CACHE BOOL "")
set(MPI_ROOT /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-upstream-2019.03.26 CACHE PATH "")
set(MPI_C_COMPILER         ${MPI_ROOT}/bin/mpicc  CACHE PATH "")
set(MPI_CXX_COMPILER       ${MPI_ROOT}/bin/mpicxx CACHE PATH "")
set(MPI_Fortran_COMPILER   /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-beta-2019.06.20/bin/mpifort CACHE PATH "")
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
set(CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -mcpu=powerpc64le -Xcompiler -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

# Uncomment this line to make nvcc output register usage for each kernel.
 set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")
