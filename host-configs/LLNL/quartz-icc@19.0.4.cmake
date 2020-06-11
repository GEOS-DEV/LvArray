set(CONFIG_NAME "quartz-icc@19.0.4" CACHE PATH "")

set(COMPILER_DIR /usr/tce/packages/intel/intel-19.0.4/compilers_and_libraries_2019.4.227/linux )
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/intel64/icc CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/intel64/icpc CACHE PATH "")
set(CMAKE_Fortran_COMPILER ${COMPILER_DIR}/bin/intel64/ifort CACHE PATH "")

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

set(MPI_HOME             /usr/tce/packages/mvapich2/mvapich2-2.3-intel-19.0.4 CACHE PATH "")

include(${CMAKE_CURRENT_LIST_DIR}/../../host-configs/LLNL/quartz-base.cmake)
