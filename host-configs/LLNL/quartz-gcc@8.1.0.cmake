set(CONFIG_NAME "quartz-gcc@8.1.0" CACHE PATH "") 

set(COMPILER_DIR /usr/tce/packages/gcc/gcc-8.1.0/ )
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/gcc CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/g++ CACHE PATH "")
set(CMAKE_Fortran_COMPILER ${COMPILER_DIR}/bin/gfortran CACHE PATH "")

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

set(MPI_HOME             /usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.1.0 CACHE PATH "")

include(${CMAKE_CURRENT_LIST_DIR}/../../host-configs/LLNL/quartz-base.cmake)
