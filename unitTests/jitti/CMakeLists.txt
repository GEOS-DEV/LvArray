get_target_property( lvarray_ITERFACE_INCLUDES lvarray INTERFACE_INCLUDE_DIRECTORIES )
get_target_property( lvarray_ITERFACE_SYSTEM_INCLUDES lvarray INTERFACE_SYSTEM_INCLUDE_DIRECTORIES )

# Logic for testDynamicLibrary. We want to compile the libraries as if they depended on lvarray.
# However we don't want to actually link against (lvarray) and associated libraries because then
# when it gets dlopen'd by testDynamicLibrary there will be duplicates. This seems to be the only way
# I could get it to work but hopefully there's a better way.
blt_add_library( NAME testDynamicLibrarySerial
                 SOURCES testDynamicLibrarySerial.cpp
                 OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                 SHARED ON )

target_include_directories( testDynamicLibrarySerial PUBLIC ${lvarray_ITERFACE_INCLUDES} )
target_include_directories( testDynamicLibrarySerial SYSTEM PUBLIC ${lvarray_ITERFACE_SYSTEM_INCLUDES} )
set_target_properties( testDynamicLibrarySerial PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set( testDynamicLibraryDependencies testDynamicLibrarySerial )

if( ENABLE_OPENMP )
  blt_add_library( NAME testDynamicLibraryOpenMP
                   SOURCES testDynamicLibraryOpenMP.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   SHARED ON )

  target_include_directories( testDynamicLibraryOpenMP PUBLIC ${lvarray_ITERFACE_INCLUDES} )
  target_include_directories( testDynamicLibraryOpenMP SYSTEM PUBLIC ${lvarray_ITERFACE_SYSTEM_INCLUDES} )
  set_target_properties( testDynamicLibraryOpenMP PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testDynamicLibraryDependencies testDynamicLibraryOpenMP )
endif()

if( ENABLE_CUDA )
  # CUDA still needs to be linked in otherwise there are some undefined reference errors when it gets dlopen'd.
  blt_add_library( NAME testDynamicLibraryCUDA
                   SOURCES testDynamicLibraryCUDA.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON cuda openmp
                   SHARED ON )

  target_include_directories( testDynamicLibraryCUDA PUBLIC ${lvarray_ITERFACE_INCLUDES} )
  target_include_directories( testDynamicLibraryCUDA SYSTEM PUBLIC ${lvarray_ITERFACE_SYSTEM_INCLUDES} )
  set_target_properties( testDynamicLibraryCUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testDynamicLibraryDependencies testDynamicLibraryCUDA )
endif()

blt_add_executable( NAME testDynamicLibrary
                    SOURCES testDynamicLibrary.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON lvarray jitti ${lvarray_dependencies} gtest )

add_dependencies( testDynamicLibrary ${testDynamicLibraryDependencies} )

blt_add_test( NAME testDynamicLibrary
              COMMAND testDynamicLibrary )

# Logic for testTypedDynamicLibrary
blt_add_library( NAME testTypedDynamicLibrarySerial
                 SOURCES testTypedDynamicLibrarySerial.cpp
                 OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                 SHARED ON )

target_include_directories( testTypedDynamicLibrarySerial PUBLIC ${lvarray_ITERFACE_INCLUDES} )
target_include_directories( testTypedDynamicLibrarySerial SYSTEM PUBLIC ${lvarray_ITERFACE_SYSTEM_INCLUDES} )
set_property( TARGET testTypedDynamicLibrarySerial PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set( testTypedDynamicLibraryDependencies testTypedDynamicLibrarySerial )

if( ENABLE_OPENMP )
  blt_add_library( NAME testTypedDynamicLibraryOpenMP
                   SOURCES testTypedDynamicLibraryOpenMP.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   SHARED ON )

  target_include_directories( testTypedDynamicLibraryOpenMP PUBLIC ${lvarray_ITERFACE_INCLUDES} )
  target_include_directories( testTypedDynamicLibraryOpenMP SYSTEM PUBLIC ${lvarray_ITERFACE_SYSTEM_INCLUDES} )
  set_property( TARGET testTypedDynamicLibraryOpenMP PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testTypedDynamicLibraryDependencies testTypedDynamicLibraryOpenMP )
endif()

if( ENABLE_CUDA )
  blt_add_library( NAME testTypedDynamicLibraryCUDA
                   SOURCES testTypedDynamicLibraryCUDA.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON cuda openmp
                   SHARED ON )

  target_include_directories( testTypedDynamicLibraryCUDA PUBLIC ${lvarray_ITERFACE_INCLUDES} )
  target_include_directories( testTypedDynamicLibraryCUDA SYSTEM PUBLIC ${lvarray_ITERFACE_SYSTEM_INCLUDES} )
  set_property( TARGET testTypedDynamicLibraryCUDA PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testTypedDynamicLibraryDependencies testTypedDynamicLibraryCUDA )
endif()

blt_add_executable( NAME testTypedDynamicLibrary
                    SOURCES testTypedDynamicLibrary.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON lvarray ${lvarray_dependencies} jitti gtest )

add_dependencies( testTypedDynamicLibrary ${testTypedDynamicLibraryDependencies} )

blt_add_test( NAME testTypedDynamicLibrary
              COMMAND testTypedDynamicLibrary )

# Logic for testTemplateCompiler
# The first step is to compile the library that will hold the compilation info. It doesn't have to be in a
# separate library from the test executable but it makes the dependency on the generated header cleaner.
blt_add_library( NAME squareAllJIT
                 SOURCES squareAllJIT.cpp
                 HEADERS squareAllJIT.hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                 DEPENDS_ON lvarray jitti ${lvarray_dependencies}
                 SHARED ON )

# The second set is to create the generated header. It parses the output compile_commands.json file to get the
# command used to build squareAllJIT.cpp. It adapts this to compile the dynamic library. However we need to manually
# specify the link commands. As with above we don't want to link to anything except if we're using cuda.
# The CMake logic is actually simpler than the above cases because it naturally pulls out all the includes and 
# by default doesn't link to anything.
set( linkDirectories "" )
set( linkLibraries "" )

if( ENABLE_CUDA )
    list( APPEND linkDirectories ${CUDA_TOOLKIT_ROOT_DIR}/lib64 )
    list( APPEND linkLibraries cudart_static cudadevrt )
endif()

add_custom_command( OUTPUT ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../../src/jitti/generateCompileCommandsHeader.py
                                   ${CMAKE_BINARY_DIR}/compile_commands.json
                                   --cpp ${CMAKE_CURRENT_LIST_DIR}/squareAllJIT.cpp
                                   --hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                                   --linker ${CMAKE_CXX_COMPILER}
                                   --linkDirectories ${linkDirectories}
                                   --linkLibraries ${linkLibraries} )

blt_add_executable( NAME testTemplateCompiler
                    SOURCES testTemplateCompiler.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON squareAllJIT gtest ${lvarray_dependencies} )

blt_add_test( NAME testTemplateCompiler
              COMMAND testTemplateCompiler )
