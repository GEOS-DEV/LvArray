
blt_add_library( NAME squareAll
                 SOURCES squareAll.cpp 
                 HEADERS squareAll.hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                 DEPENDS_ON lvarray jitti raja openmp 
                 SHARED ON )

get_target_property( squareAll_LINK_LIBRARIES squareAll LINK_LIBRARIES )

add_custom_command( OUTPUT ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../../src/jitti/generateCompileCommandsHeader.py
                                   ${CMAKE_BINARY_DIR}/compile_commands.json
                                   --cpp ${CMAKE_CURRENT_LIST_DIR}/squareAll.cpp
                                   --hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                                   --linker ${CMAKE_CXX_COMPILER}
                                   --linkDirectories ${CMAKE_BINARY_DIR}/lib
                                   --linkLibraries ${squareAll_LINK_LIBRARIES}
                                
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

blt_print_target_properties(TARGET squareAll)

blt_add_executable( NAME testSquareAllJIT
                    SOURCES testSquareAllJIT.cpp squareAll.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON squareAll gtest )





blt_add_test( NAME testSquareAllJIT
              COMMAND testSquareAllJIT )
