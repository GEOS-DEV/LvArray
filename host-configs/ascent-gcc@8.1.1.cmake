set(CONFIG_NAME "ascent-gcc@8.1.1" CACHE PATH "") 

# Set up the tpls
set(GEOSX_TPL_ROOT_DIR "/ccsopen/home/klevtsov/thirdPartyLibs" CACHE PATH "")
set(GEOSX_TPL_DIR ${GEOSX_TPL_ROOT_DIR}/install-${CONFIG_NAME}-release CACHE PATH "")

# C options
set(CMAKE_C_COMPILER "/sw/ascent/gcc/8.1.1/bin/gcc" CACHE PATH "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g ${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

# C++ options
set(CMAKE_CXX_COMPILER "/sw/ascent/gcc/8.1.1/bin/g++" CACHE PATH "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le -Wno-strict-aliasing" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wno-strict-aliasing" CACHE STRING "")
set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

# Fortran options
set(CMAKE_Fortran_COMPILER "/sw/ascent/gcc/8.1.1/bin/gfortran" CACHE PATH "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=power9 -mtune=power9" CACHE STRING "")
#set(FORTRAN_MANGLE_NO_UNDERSCORE ON CACHE BOOL "" FORCE)

# OpenMP options
#set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)
#set(OpenMP_Fortran_FLAGS "-fopenmp" CACHE STRING "")
#set(OpenMP_Fortran_LIB_NAMES "" CACHE STRING "")

# MPI options
set(ENABLE_MPI ON CACHE BOOL "")
set(MPI_ROOT "/autofs/nccsopen-svm1_sw/ascent/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-8.1.1/spectrum-mpi-10.3.0.0-20190419-juwhiawazufuuoastohlsuewaklwmazi" CACHE PATH "")
set(MPI_C_COMPILER         "${MPI_ROOT}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER       "${MPI_ROOT}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER   "${MPI_ROOT}/bin/mpifort" CACHE PATH "")
set(MPIEXEC                "jsrun -c1 -g1" CACHE STRING "")
set(MPIEXEC_NUMPROC_FLAG   -n CACHE STRING "")
set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC ON CACHE BOOL "")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR "/sw/ascent/cuda/10.1.105" CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER} CACHE STRING "")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set(CUDA_ARCH sm_70 CACHE STRING "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -mcpu=powerpc64le -Xcompiler -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

# Uncomment this line to make nvcc output register usage for each kernel.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")
