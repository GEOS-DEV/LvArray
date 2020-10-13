###############################################################################
Build Guide
###############################################################################

LvArray uses a CMake based build system augmented with `BLT <https://github.com/LLNL/blt>`_. If you're not familiar with CMake, RAJA has a good `introduction <https://raja.readthedocs.io/en/main/getting_started.html#getting-started-label>`_. LvArray has a dependency on RAJA and as such requires that the ``RAJA_DIR`` CMake variable defined and points to a RAJA installation.

CMake Options
-------------
In addition to the standard CMake and BLT options LvArray supports the following options.

* **Adding additional targets**
    The following variables add additional build targets but do not alter the usage or functionality of LvArray.

    * ``ENABLE_TESTS`` : default ``ON``
        Build the unit tests which can be run with ``make test``. The unit tests take a long time to build with the IMB XL and Intel compilers.
    * ``ENABLE_EXAMPLES`` : default ``ON``
        Build the examples, ``ENABLE_TESTS`` must also be ``ON``.
    * ``ENABLE_BENCHMARKS`` : default ``ON``
        Build the benchmarks, ``ENABLE_TESTS`` must also be ``ON``.
    * ``DISABLE_UNIT_TESTS`` : default ``OFF``
        Use with ``ENABLE_TESTS=ON`` to disable building the unit tests but still allow the examples and benchmarks.
    * ``ENABLE_DOCS`` : default ``ON``
        Build the documentation.

* **Third party libraries**
    LvArray has a hard dependency on RAJA along with multiple optional dependencies. Of the following variables only ``RAJA_DIR`` is mandatory.

    * ``RAJA_DIR``
        The path to the RAJA installation.
    * ``ENABLE_UMPIRE`` : default ``OFF``
        If Umpire is enabled. Currently no part of LvArray uses Umpire but it is required when using CHAI.
    * ``UMPIRE_DIR``
        The path to the Umpire installation, must be specified when Umpire is enabled.
    * ``ENABLE_CHAI`` : default ``OFF``
        If CHAI is enabled, CHAI also requires Umpire. Enabling CHAI allows the usage of the ``LvArray::ChaiBuffer``.
    * ``CHAI_DIR``
        The path to the CHAI installation, must be specified when CHAI is enabled.
    * ``ENABLE_CALIPER`` : default ``OFF``
        If caliper is enabled. Currently caliper is only used to time the benchmarks.
    * ``CALIPER_DIR``
        The path to the caliper installation, must be specified when caliper is enabled.
    * ``ENABLE_ADDR2LINE`` : default ``ON``
        If ``addr2line`` is enabled. This is used in ``LvArray::system::stackTrace`` to attempt to provide file and line locations for the stack frames.
    * ``ADDR2LINE_EXEC`` : default ``/usr/bin/addr2line``
        The path to the ``addr2line`` executable.

* **Debug options**
    The following options don't change the usage of LvArray but they are intended to make debugging easier.

    * ``LVARRAY_BOUNDS_CHECK`` : default ``ON`` iff in a CMake ``Debug`` build.
        Enables bounds checks on container access along with checks for other invalid operations.
    * ``ENABLE_TOTALVIEW_OUTPUT`` : default ``OFF``
        Makes it easier to inspect the ``LvArray::Array`` in TotalView. This functionality is highly dependent on the version of TotalView used.

Using LvArray Your Application
------------------------------
Once LvArray has been installed if your application uses CMake importing LvArray is as simple as defining ``LVARRAY_DIR`` as the path to LvArray install directory and then adding ``find_package(LVARRAY)``. This will export a ``lvarray`` target that can then be used by ``target_link_libraries`` and the like.

Host Configs
------------
Host config files are a convenient way to group CMake options for a specific configuration together. There are a set of example host configs in the ``host-configs`` directory. Once you've created a host config file you can use ``scripts/config-build.py`` to create the build directory and run CMake for you. An example usage would be ``python ./scripts/config-build.py --hc host-configs/LLNL/quartz-clang@10.0.0.cmake``.

::

    > python scripts/config-build.py --help
    usage: config-build.py [-h] [-bp BUILDPATH] [-ip INSTALLPATH]
                          [-bt {Release,Debug,RelWithDebInfo,MinSizeRel}] [-e]
                          [-x] [-ecc] -hc HOSTCONFIG

    Configure cmake build. Unrecognized arguments are passed on to CMake.

    optional arguments:
      -h, --help            show this help message and exit
      -bp BUILDPATH, --buildpath BUILDPATH
                            specify path for build directory. If not specified,
                            will create in current directory.
      -ip INSTALLPATH, --installpath INSTALLPATH
                            specify path for installation directory. If not
                            specified, will create in current directory.
      -bt {Release,Debug,RelWithDebInfo,MinSizeRel}, --buildtype {Release,Debug,RelWithDebInfo,MinSizeRel}
                            build type.
      -e, --eclipse         create an eclipse project file.
      -x, --xcode           create an xcode project.
      -ecc, --exportcompilercommands
                            generate a compilation database. Can be used by the
                            clang tools such as clang-modernize. Will create a
                            file called 'compile_commands.json' in build
                            directory.
      -hc HOSTCONFIG, --hostconfig HOSTCONFIG
                            select a specific host-config file to initalize
                            CMake's cache

Submodule usage
---------------
LvArray can also be used as a submodule. In this case the configuration is largely the same except that LvArray expects the parent project to have imported the third party libraries. For example if ``ENABLE_UMPIRE`` is ``ON`` then LvArray will depend on ``umpire`` but it will make no attempt to find these library (``UMPIRE_DIR`` is unused).

Spack and Uberenv Builds
------------------------
LvArray has an associated `Spack <https://github.com/spack/spack>`_ package. For those unfamiliar with Spack the most important thing to understand is the `spec syntax <https://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_. For those interested the LvArray package implementation is `here <https://github.com/corbett5/spack/blob/feature/corbett/lvarray/var/spack/repos/builtin/packages/lvarray/package.py>`_ the important part of which is reproduced below.

.. code:: python

    class Lvarray(CMakePackage, CudaPackage):
        """LvArray portable HPC containers."""

        homepage = "https://github.com/GEOSX/lvarray"
        git      = "https://github.com/GEOSX/LvArray.git"

        version('develop', branch='develop', submodules='True')
        version('tribol', branch='temp/feature/corbett/tribol', submodules='True')

        variant('shared', default=True, description='Build Shared Libs')
        variant('umpire', default=False, description='Build Umpire support')
        variant('chai', default=False, description='Build Chai support')
        variant('caliper', default=False, description='Build Caliper support')
        variant('tests', default=True, description='Build tests')
        variant('benchmarks', default=False, description='Build benchmarks')
        variant('examples', default=False, description='Build examples')
        variant('docs', default=False, description='Build docs')
        variant('addr2line', default=True,
                description='Build support for addr2line.')

        depends_on('cmake@3.8:', type='build')
        depends_on('cmake@3.9:', when='+cuda', type='build')

        depends_on('raja')
        depends_on('raja+cuda', when='+cuda')

        depends_on('umpire', when='+umpire')
        depends_on('umpire+cuda', when='+umpire+cuda')

        depends_on('chai+raja', when='+chai')
        depends_on('chai+raja+cuda', when='+chai+cuda')

        depends_on('caliper', when='+caliper')

        depends_on('doxygen@1.8.13:', when='+docs', type='build')
        depends_on('py-sphinx@1.6.3:', when='+docs', type='build')


LvArray also has an ``uberenv`` based build which simplifies building LvArray's dependencies along with optionally LvArray using spack.

::

    > ./scripts/uberenv/uberenv.py --help
    Usage: uberenv.py [options]

    Options:
      -h, --help            show this help message and exit
      --install             Install `package_name`, not just its dependencies.
      --prefix=PREFIX       destination directory
      --spec=SPEC           spack compiler spec
      --mirror=MIRROR       spack mirror directory
      --create-mirror       Create spack mirror
      --upstream=UPSTREAM   add an external spack instance as upstream
      --spack-config-dir=SPACK_CONFIG_DIR
                            dir with spack settings files (compilers.yaml,
                            packages.yaml, etc)
      --package-name=PACKAGE_NAME
                            override the default package name
      --package-final-phase=PACKAGE_FINAL_PHASE
                            override the default phase after which spack should
                            stop
      --package-source-dir=PACKAGE_SOURCE_DIR
                            override the default source dir spack should use
      --project-json=PROJECT_JSON
                            uberenv project settings json file
      -k                    Ignore SSL Errors
      --pull                Pull if spack repo already exists
      --clean               Force uninstall of packages specified in project.json
      --run_tests           Invoke build tests during spack install
      --macos-sdk-env-setup
                            Set several env vars to select OSX SDK settings.This
                            was necessary for older versions of macOS  but can
                            cause issues with macOS versions >= 10.13.  so it is
                            disabled by default.

Two simple examples are provided below.

::

    quartz2498 > ./scripts/uberenv/uberenv.py --install --spec="@develop %clang@10.0.1"

This will build RAJA (LvArray's only hard dependency) and LvArray and install them in ``./uberenv_libs/linux-rhel7-ppc64le-clang@10.0.1``. By default libraries are built in the ``RelWithDebInfo`` CMake configuration.

::

    quartz2498 > ./scripts/uberenv/uberenv.py --spec="@develop %gcc@8.3.1 ^raja@0.12.1 build_type=Release"

This will install RAJA in the same location but it will be built in the ``Release`` configuration and instead of building and installing LvArray a host-config will be generated and placed in the current directory. This can be useful for developing or debugging.

Currently ``uberenv`` only works on the LLNL ``toss_3_x86_64_ib`` and ``blueos_3_ppc64le_ib_p9`` systems. Further more only certain compilers are supported. On the TOSS systems ``clang@10.0.1``, ``gcc@8.3.1`` and ``intel@19.1.2`` are supported. On BlueOS ``clang-upstream-2019.08.15 (clang@9.0.0)``, ``clang-ibm-10.0.1-gcc-8.3.1 (clang@10.0.1)``, ``gcc@8.3.1`` and ``xl-2020.09.17-cuda-11.0.2 (xl@16.1.1)`` are supported. Adding support for more compilers is as simple as adding them to the appropriate ``compilers.yaml`` file.

Adding support for a new system is easy too, you just need to create a directory with a ``compilers.yaml`` which specifies the available compilers and a ``packages.yaml`` for system packages and then pass this directory to uberenv with the ``--spack-config-dir`` option.

For reference two more complicated specs are shown below

::

  lassen709 > ./scripts/uberenv/uberenv.py --install --run_tests --spec="@develop+umpire+chai+caliper+cuda %clang@10.0.1 cuda_arch=70 ^cuda@11.0.2 ^raja@0.12.1~examples~exercises cuda_arch=70 ^umpire@4.0.1~examples cuda_arch=70 ^chai@master~benchmarks~examples cuda_arch=70 ^caliper@2.4~adiak~mpi~dyninst~callpath~papi~libpfm~gotcha~sampler~sosflow"

This will use ``clang@10.0.1`` and ``cuda@11.0.2`` to build and install RAJA v0.12.1 without examples or exercises, Umpire v4.0.1 without examples, the master branch of CHAI without benchmarks or examples, and caliper v2.4 without a bunch of options. Finally it will build and install LvArray after running the unit tests and verifying that they pass. Note that each package that depends on cuda gets the ``cuda_arch=70`` variable.

::

    quartz2498 > ./scripts/uberenv/uberenv.py --spec="@tribol+umpire %intel@19.1.2 ^raja@0.12.1 build_type=Release ^umpire@4.0.1 build_type=Release"

This will use ``intel@19.1.2`` to build and install RAJA V0.12.1 in release and Umpire v4.0.1 in release. Finally it will generate a host config that can be used to build LvArray.
