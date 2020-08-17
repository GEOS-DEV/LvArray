import json
import os
import argparse
import sys
import re
import csv
import collections
import numpy as np


class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    RESET = '\033[0m'

def getBenchmarksFromJsonFile( jsonFile ):
    """
    Return an ordered dictionary containing the benchmarks from the benchmark file.
    
    The keys are the benchmark names and the values are lists of benchmark repetitions.

    Arguments:
        jsonFile: The google benchmark json output file.
    """
    with open( jsonFile, "r") as f:
        file = json.load( f )

    timeUnit = None
    benchmarks = collections.OrderedDict()
    for benchmark in file[ "benchmarks" ]:
        if benchmark[ "run_type" ] == "iteration":
            if timeUnit is None:
                timeUnit = benchmark[ "time_unit" ]
            elif timeUnit != benchmark[ "time_unit" ]:
                raise Exception( "Time unit mismatch!" )

            benchmarks.setdefault( benchmark[ "name" ], [] ).append( benchmark )

    return benchmarks


def computeStatistic( benchmarks, field, func ):
    """
    Return the result of func applied to the values of field in benchmarks.
    
    Arguments:
        benchmarks: The list of benchmarks to gather data from.
        field: The field to gather from the benchmarks.
        func: The function to apply to the data, must accept a list and return a single value.
    """
    results = []
    for benchmark in benchmarks:
        results.append( benchmark[ field ] )

    return func( results )


def getResults( benchmarks, field, func, baselineRegex, resultsPattern ):
    """
    Return an ordered dictionary containing the baseline benchmarks and the benchmarks to compare them against.
    
    The keys of the dictionary are the baseline benchmark names, the values are themselves an ordered dictionary. 
    The keys of the inner dictionary are the names of the matched benchmarks and the values are their computed statistic.
    Further more each inner dictionary contains two special key-value pairs, 'baseline' which contains the statistics
    of the baseline benchmark, and 'name' which contains the shortened name of the baseline benchmark.

    Arguments:
        benchmarks: The list of benchmarks to gather data from.
        field: The field to gather from the benchmarks.
        func: The function to apply to the data, must accept a list and return a single value.
        baselineRegex: The regex used to match the baseline benchmarks.
        resultsPattern: The pattern used to construct the regex to match against the
            runs to compare against each baseline.
    """
    results = collections.OrderedDict()
    for benchmarkName in benchmarks:
        if re.match( baselineRegex, benchmarkName ) is not None:
            results[ benchmarkName ] = collections.OrderedDict()
            results[ benchmarkName ][ "baseline" ] = computeStatistic( benchmarks[ benchmarkName ], field, func )
            results[ benchmarkName ][ "name" ] = benchmarkName[: benchmarkName.find( '/' ) ]
    
    for baselineName in results:
        groups = re.match( baselineRegex, baselineName ).groups()
        resultsRegex = resultsPattern.format( *groups )

        for benchmarkName in benchmarks:
            if benchmarkName == baselineName:
                continue

            match = re.match( resultsRegex, benchmarkName )
            if match is not None:
                groups = match.groups()
                results[ baselineName ][ " ".join( groups ) ] = computeStatistic( benchmarks[ benchmarkName ], field, func )

    return results


def toString( x ):
    """
    Return a string representing x. If x is a float convert it to scientific notation.

    Arguments:
        x: The value to convert to a string.
    """
    if isinstance( x, float ):
        return "{:.2e}".format( x )
    else:
        return str( x )


def getValue( x ):
    """
    If x is a tuple return the first entry, else return x.

    Arguments:
        x: The object to get the value of.
    """
    if isinstance( x, tuple ):
        return x[ 0 ]
    else:
        return x


def getColor( x ):
    """
    If x is a tuple return the second entry, which should be an ANSI color code. Else return the default color.

    Arguments:
        x: The object to get the color of.
    """
    if isinstance( x, tuple ):
        return x[ 1 ]
    else:
        return style.RESET


def printTable( table ):
    """
    Print a table in a nice format, with optional coloring.

    Arguments:
        table: A list of rows to print. Each row should be of the same length. Then entries in each row
            should either be a string or a tuple of a string and ANSI color code.
    """
    col_width = [ max( len( getValue( x ) ) for x in col ) for col in zip( *table ) ]
    print( "| " + " | ".join( "{:{}}".format( getValue( x ), col_width[ i ] ) for i, x in enumerate( table[ 0 ] ) ) + " |" )
    print( "|" + "|".join( "-" * width + "--" for width in col_width ) + "|" )

    for line in table[ 1: ]:
        print( "| " + " | ".join( "{}{:{}}{}".format( getColor( x ), getValue( x ), col_width[ i ], style.RESET ) for i, x in enumerate( line ) ) + " |" )

    print( "|" + "|".join( "-" * width + "--" for width in col_width ) + "|" )


def aggregateAndPrint( results ):
    """
    Print an ordered dictionary of results as produced by getResults.

    Arguments:
        results: The ordered dictionary to print.
    """
    colors = {}
    resultNames = []
    for groupName in results:
        group = results[ groupName ]
        colors[ groupName ] = {}
        for name in group:
            if name not in ( "baseline", "name" ):
                if name not in resultNames:
                    resultNames.append( name )

                value = group[ name ] / group[ "baseline" ]
                if value > 1.05:
                    colors[ groupName ][ name ] = style.GREEN
                elif value < 0.95:
                    colors[ groupName ][ name ] = style.RED
                else:
                    colors[ groupName ][ name ] = style.RESET

                group[ name ] = "{:.4}x".format( value )

    columnNames = [ "name", "baseline" ] + resultNames
    rowNames = results.keys()

    table = []
    table.append( columnNames )
    for rowName in rowNames:
        row = []
        for columnName in columnNames:
            value = toString( results[ rowName ].get( columnName, float( "nan" ) ) )
            color = colors[ rowName ].get( columnName, style.RESET )
            row.append( ( value, color ) )

        table.append( row )

    printTable( table )

    # writer = csv.DictWriter( sys.stdout, columnNames, restval=float( "nan" ) )
    # writer.writeheader()
    
    # for rowName in rowNames:
    #     writer.writerow( results[ rowName ] )


def compare( jsonFile, baselineRegex, resultsPattern, field, func ):
    """
    Compare the benchmarks from the given file.

    Arguments:
        jsonFile: The json google benchmark output file.
        baselineRegex: The regex used to match the baseline benchmarks.
        resultsPattern: The pattern used to construct the regex to match against the
            runs to compare against each baseline.
        field: The field of each benchmark to compare.
        func: A function that reduces a list of values into a single value.
            Used to aggregate benchmark results.
    """
    if not os.path.isfile( jsonFile ):
        raise ValueError( "jsonFile is not a file!" )
    
    benchmarks = getBenchmarksFromJsonFile( jsonFile )

    results = getResults( benchmarks, field, func, baselineRegex, resultsPattern )
    aggregateAndPrint( results )


def main():
    """ Parse the command line arguments and compare the benchmarks. """

    parser = argparse.ArgumentParser( description="Compare benchmarks within a google benchmark suite.",
                                      epilog="Assuming the standard Array benchmark nomenclature to compare the pointer implementation against the various "
                                             "abstractions you would have a baseline regex of 'pointer(.*)' and a results pattern of '(.*){}'. If you wanted to "
                                             " compare the native implementation vs the various RAJA implementations you would have a baseline regex of "
                                             "'(.*)Native<(.*)>' and a results pattern of '{0}Raja<std::pair< {1}, (.*) >>.*'." )
    parser.add_argument( "benchmarkFile", help="The path to the benchmark results json file.")
    parser.add_argument( "baselineRegex", help="The regex used to match the baseline benchmarks, each match generates a new row in the output table. "
                                               "The regex may optionally contain one or more match groups, which are then used to format the results pattern to yield the results regex." )
    parser.add_argument( "resultsPattern", help="The pattern used construct the regex used to match against the result benchmarks, each match generates a new column. "
                                                "For each baseline benchmark the pattern is formatted with the match groups from the baseline regex and then matched against the "
                                                "remaining benchmarks.For each match any resulting match groups are used to name the column." )
    parser.add_argument( "field", help="The field to compare." )
    parser.add_argument( "func", help="The function used to reduce the results. This is a string that is eval'd to produce a function that takes an array and returns a number. "
                                      "This can be as simple as 'sum', 'min', 'np.mean', or more complicated like 'lambda x: np.sum( np.array( x ) ** 2 )'" )
    args = parser.parse_args()

    print( "Benchmark file: {}".format( args.benchmarkFile ) )
    print( "Baseline regex: {}".format( args.baselineRegex ) )
    print( "Results pattern: {}".format( args.resultsPattern ) )
    print( "Field to compare: {}".format( args.field ) )
    print( "Reduction method: {}".format( args.func ) )

    compare( args.benchmarkFile, args.baselineRegex, args.resultsPattern, args.field, eval( args.func ) )
    return 0

if __name__ == "__main__" and not sys.flags.interactive:
    sys.exit(main())
