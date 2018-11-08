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

set( ENABLE_CHAI ON CACHE BOOL "" FORCE )
set( CHAI_DIR "/usr/gapps/GEOS/geosx/thirdPartyLibs/install-toss_3_x86_64_ib-clang@4.0.0-release/chai" CACHE PATH "" FORCE )

set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-9.2.148" CACHE STRING "")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set (CUDA_ARCH "sm_60" CACHE STRING "")
#set (CMAKE_CUDA_LINK_FLAGS "-Xlinker -rpath -Xlinker /usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2017.04.03/lib -Xlinker -rpath -Xlinker /usr/tce/packages/clang/clang-coral-2017.10.13/ibm/lib:/usr/tce/packages/gcc/gcc-4.9.3/lib64 -L/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2017.04.03/lib/ -lmpi_ibm" CACHE STRING "")
set (CUDA_NVCC_FLAGS -restrict -arch ${CUDA_ARCH} --expt-extended-lambda CACHE LIST "" )
#set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
