set(CONFIG_NAME "ray_blueos_3_ppc64le_ib-clang@upstream" CACHE PATH "") 

set(GEOSX_TPL_ROOT_DIR "/usr/gapps/GEOS/geosx/thirdPartyLibs" CACHE PATH "")
set(GEOSX_TPL_DIR "${GEOSX_TPL_ROOT_DIR}/install-${CONFIG_NAME}-release" CACHE PATH "")

set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-upstream-2019.03.19/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-upstream-2019.03.19/bin/clang++" CACHE PATH "")

# set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "")
# set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} CACHE STRING "")
# set(CMAKE_Fortran_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} CACHE STRING "")

set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)

# MPI options
set(ENABLE_MPI ON CACHE BOOL "")
set(MPI_ROOT "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-upstream-2019.03.19" CACHE PATH "")
set(MPI_C_COMPILER         "${MPI_ROOT}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER       "${MPI_ROOT}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER   "${MPI_ROOT}/bin/mpif90" CACHE PATH "")
set(MPIEXEC                "mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np" CACHE PATH "")
# set(BLT_MPI_COMMAND_APPEND "/usr/tcetmp/bin/mpibind" CACHE PATH "")
# set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC ON CACHE BOOL "")

# Fortran options
set(ENABLE_Fortran ON CACHE BOOL "")
set(CMAKE_Fortran_COMPILER "/usr/tcetmp/packages/gcc/gcc-4.9.3/bin/gfortran" CACHE PATH "")
set(BLT_EXE_LINKER_FLAGS "-lgfortran -Wl,-rpath,/usr/tcetmp/packages/gcc/gcc-4.9.3/lib64/" CACHE PATH "")
# set(CMAKE_Fortran_COMPILER_ID "XL" CACHE PATH "All of BlueOS compilers report clang due to nvcc, override to proper compiler family")
# set(BLT_Fortran_FLAGS "-WF,-C!" CACHE PATH "Converts C-style comments to Fortran style in preprocessed files")
# set(BLT_EXE_LINKER_FLAGS "-Wl,-rpath,/usr/tce/packages/xl/xl-2019.04.19/lib/" CACHE PATH "Adds a missing rpath for libraries associated with the fortran compiler")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-10.1.105" CACHE STRING "")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set(CUDA_ARCH "sm_60" CACHE STRING "")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE LIST "" )
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER} CACHE STRING "")
set(BLT_CUDA_FLAGS "${BLT_CUDA_FLAGS} ${CUDA_NVCC_FLAGS}" CACHE STRING "")

set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")
