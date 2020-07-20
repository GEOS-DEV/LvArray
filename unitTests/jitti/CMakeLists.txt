
add_custom_command( OUTPUT ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../../src/jitti/generateCompileCommandsHeader.py
                                   ${CMAKE_BINARY_DIR}/compile_commands.json
                                   ${CMAKE_CURRENT_LIST_DIR}/squareAll.cpp
                                   ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

blt_add_library( NAME squareAllDummy
                 SOURCES squareAll.cpp 
                 HEADERS squareAll.hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                 HEADERS squareAll.hpp
                 DEPENDS_ON lvarray ${lvarray_dependencies} jitti
                 SHARED ON )

blt_add_executable( NAME testSquareAllJIT
                    SOURCES testSquareAllJIT.cpp squareAll.cpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON lvarray ${lvarray_dependencies} jitti gtest )


blt_add_test( NAME testSquareAllJIT
              COMMAND testSquareAllJIT )
