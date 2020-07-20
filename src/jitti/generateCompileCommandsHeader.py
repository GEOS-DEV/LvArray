import sys
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "compileCommandsPath", help="The path to the compile_commands.json file." )
    parser.add_argument( "cppFilePath", help="The translation unit to get the compilation arguments of." )
    parser.add_argument( "headerFilePath", help="The path of the header file to generate." )
    args = parser.parse_args()

    compileCommandsPath = os.path.abspath( args.compileCommandsPath )
    if not os.path.isfile( compileCommandsPath ):
        raise Exception( "{} could not be found".format( compileCommandsPath ) )

    cppFilePath = os.path.abspath( args.cppFilePath )
    if not os.path.isfile( cppFilePath ):
        raise Exception( "{} could not be found".format( cppFilePath ) )

    cppFileName = os.path.splitext( os.path.basename( cppFilePath ) )[ 0 ]

    headerFilePath = os.path.abspath( args.headerFilePath )


    command = None
    with open( compileCommandsPath, "r" ) as compileCommands:
        translationUnits = json.load( compileCommands )

        for tu in translationUnits:
            if tu[ "file" ] == cppFilePath:
                command = tu[ "command" ]

    if command is None:
        raise Exception( "Could not find {} in {}".format( cppFilePath, compileCommandsPath ) )

    # Remove everything after "-c or -o"
    pos = min( command.find( " -c " ), command.find( " -o " ) )
    if pos < 0:
        raise Exception( "Could not reformat the compilation command: {}".format( command ) )

    command = command[ 0 : pos ]

    includeGuard = "GUARD_" + str( abs( hash( command ) ) )

    newHeaderContents = ( "#ifndef {}\n".format( includeGuard ) +
                          "#define {}\n\n".format( includeGuard ) +
                          "#define {}_COMPILE_COMMANDS \"{}\"\n\n".format( cppFileName, command ) +
                          "#endif\n" )

    # If the header file exists check to see if it needs to be modified.
    # If it is modified then all dependent targets get rebuild so it should
    # only be done if necessary.
    # if os.path.isfile( headerFilePath ):
    #     with open( headerFilePath, "r" ) as existingHeader:
    #         if existingHeader.read() == newHeaderContents:
    #             return 0

    with open( headerFilePath, "w" ) as output:
        output.write( newHeaderContents )

    return 0

if __name__ == "__main__":
    sys.exit( main() )
