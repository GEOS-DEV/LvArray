import sys
import os
import json
import argparse


def getCompileCommand( compileCommandsPath, cppFilePath ):
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

    return command[ 0 : pos ]


def getLinkArgs( linkDirectories, linkLibraries ):
    print( linkDirectories )
    print( linkLibraries )

    command = []
    for linkDir in linkDirectories:
        command.append( "-L" + linkDir )

    for arg in linkLibraries:
        libs = arg.split( " " )
        for lib in libs:
            if lib.endswith( ".a" ) or lib.endswith( ".so" ):
                command.append( lib )
            elif lib.startswith("-"):
                command.append( lib )
            else:
                command.append( "-l" + lib )

    print(command)
    return " ".join( command )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "compileCommandsPath", type=str,
                         help="The path to the compile_commands.json file." )
    parser.add_argument( "--cpp", dest="cppFilePath", required=True, type=str,
                         help="The translation unit to get the compilation arguments of." )
    parser.add_argument( "--hpp", dest="headerFilePath", required=True, type=str,
                         help="The path of the header file to generate." )
    parser.add_argument( "--linker", dest="linker", required=True, type=str,
                         help="The program to use for linking" )
    parser.add_argument( "--includeDirectories", dest="includeDirectories", required=False, type=str, nargs="*",
                         default=[], help="Extra directories to include." )
    parser.add_argument( "--linkDirectories", dest="linkDirectories", required=False, type=str, nargs="*",
                         default=[], help="The directories to add to the library search path." )
    parser.add_argument( "--linkLibraries", dest="linkLibraries", required=False, type=str, nargs="*",
                         default=[], help="The libraries to link against")
    args = parser.parse_args()

    compileCommandsPath = os.path.abspath( args.compileCommandsPath )
    if not os.path.isfile( compileCommandsPath ):
        raise Exception( "{} could not be found".format( compileCommandsPath ) )

    cppFilePath = os.path.abspath( args.cppFilePath )
    if not os.path.isfile( cppFilePath ):
        raise Exception( "{} could not be found".format( cppFilePath ) )

    cppFileName = os.path.splitext( os.path.basename( cppFilePath ) )[ 0 ]

    headerFilePath = os.path.abspath( args.headerFilePath )

    compileCommand = getCompileCommand( compileCommandsPath, cppFilePath )
    for includeDir in args.includeDirectories:
        compileCommand += " -I" + includeDir

    linkCommand = getLinkArgs( args.linkDirectories, args.linkLibraries )

    includeGuard = "GUARD_" + str( abs( hash( compileCommand ) ) )

    newHeaderContents = ( "#ifndef {}\n".format( includeGuard ) +
                          "#define {}\n\n".format( includeGuard ) +
                          "#define {}_COMPILE_COMMAND \"{}\"\n\n".format( cppFileName, compileCommand ) +
                          "#define {}_LINKER \"{}\"\n\n".format( cppFileName, args.linker ) +
                          "#define {}_LINK_ARGS \"{}\"\n\n".format( cppFileName, linkCommand ) +
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
