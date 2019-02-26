set( cxx_dependencies "" )
if( ATK_FOUND OR ENABLE_ATK OR ENABLE_AXOM )
    set( cxx_dependencies slic lumberjack )
endif()

if(ENABLE_CHAI)
    set( cxx_dependencies ${cxx_dependencies} chai umpire )
endif()

if(ENABLE_MPI)
    set( cxx_dependencies ${cxx_dependencies} mpi )
endif()

if(ENABLE_CUDA)
    set( cxx_dependencies ${cxx_dependencies} cuda )
endif()

add_subdirectory(src)
add_subdirectory(tests)
