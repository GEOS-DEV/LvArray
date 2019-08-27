set( cxx_dependencies "" )
if( ATK_FOUND )
    set( cxx_dependencies axom )
endif()

if( ENABLE_CHAI )
    set( cxx_dependencies ${cxx_dependencies} chai umpire )
endif()

if( ENABLE_MPI )
    set( cxx_dependencies ${cxx_dependencies} mpi )
endif()

if( ENABLE_CUDA )
    set( cxx_dependencies ${cxx_dependencies} cuda )
endif()

if( ENABLE_OPENMP )
    set( cxx_dependencies ${cxx_dependencies} openmp )
endif()

if( ENABLE_RAJA )
    set( cxx_dependencies ${cxx_dependencies} raja )
endif()

if( ENABLE_CALIPER )
    set( cxx_dependencies ${cxx_dependencies} caliper )
endif()

add_subdirectory(src)
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
