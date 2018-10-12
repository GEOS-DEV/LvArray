set(CONFIG_NAME "blueos_3_ppc64le_ib-clang@coral" CACHE PATH "") 


set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-coral-2018.08.08/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-coral-2018.08.08/bin/clang++" CACHE PATH "")

set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)



# fortran compiler used by spack
set(ENABLE_FORTRAN OFF CACHE BOOL "")

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "")

set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_HOME                 "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-coral-2018.08.08" CACHE PATH "")

set(MPI_C_COMPILER           "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER         "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER     "${MPI_HOME}/bin/mpifort" CACHE PATH "")

set(MPIEXEC                "${MPI_HOME}/bin/mpirun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG   "-np" CACHE PATH "")
set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE PATH "")


set(ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-9.2.148" CACHE BOOL "" FORCE)

set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
