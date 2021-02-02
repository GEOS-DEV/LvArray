#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
project_dir="$(pwd)"

hostconfig=${HOST_CONFIG:-""}

name=${NAME}

spec=${SPEC:-""}

build_type=${BUILD_TYPE}

# Dependencies
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    prefix_opt=""

    if [[ -d /dev/shm ]]
    then
        prefix="/dev/shm/${hostname}/${name}"
        mkdir -p ${prefix}
        prefix_opt="--prefix=${prefix}"
    fi

    python scripts/uberenv/uberenv.py --spec="${spec}" ${prefix_opt}
fi

# Host config file
# We are looking for a unique host config in project dir.
hostconfigs=( $( ls "${project_dir}/"*@*.cmake ) )
if [[ ${#hostconfigs[@]} == 1 ]]
then
    hostconfig_path=${hostconfigs[0]}
    echo "Found host config file: ${hostconfig_path}"
elif [[ ${#hostconfigs[@]} == 0 ]]
then
    echo "No result for: ${project_dir}/hc-*.cmake"
    echo "Spack generated host-config not found."
    exit 1
else
    echo "More than one result for: ${project_dir}/hc-*.cmake"
    echo "${hostconfigs[@]}"
    echo "Please specify one with HOST_CONFIG variable"
    exit 1
fi


build_dir="${project_dir}/build_"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Project Dir: ${project_dir}"
echo "~~~~~ Host-config: ${hostconfig_path}"
echo "~~~~~ Build Dir:   ${build_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Build
for buildType in ${build_type}
do
    cd ${project_dir}

    if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
    then
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~ Building LvArray"
        echo "~~~~~ Build Type: ${buildType}"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

        python scripts/config-build.py -hc ${hostconfig_path} -bt ${buildType} -bp ${build_dir}

        cd ${build_dir}
        cmake --build . -j
    fi

    # Test
    if [[ "${option}" != "--build-only" ]]
    then
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~ Testing LvArray"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

        if [[ ! -d ${build_dir} ]]
        then
            echo "ERROR: Build directory not found : ${build_dir}" && exit 1
        fi

        cd ${build_dir}

        ctest --output-on-failure -T test 2>&1 | tee tests_output.txt

        no_test_str="No tests were found!!!"
        if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
        then
            echo "ERROR: No tests were found" && exit 1
        fi

        echo "Copying Testing xml reports for export"
        tree Testing
        cp Testing/*/Test.xml ${project_dir}

        if grep -q "Errors while running CTest" ./tests_output.txt
        then
            echo "ERROR: failure(s) while running CTest" && exit 1
        fi
    fi
done
