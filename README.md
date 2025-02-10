# Arnoldi Iteration

## Setup

### Spack

This assumes that you have spack installed and sourced. If not, you can install
it using the following commands

```bash shell
$ git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git

# For bash/zsh/sh
$ . spack/share/spack/setup-env.sh
# For tcsh/csh
$ source spack/share/spack/setup-env.csh
# For fish
$ . spack/share/spack/setup-env.fish

# Then to discover available compilers
$ spack compiler list
$ spack compiler find
```

Now in the root directory of this repository, you can load the spack environment
that contains the packages needed to build the project.

```bash shell
# 1. Activate the local environment
$ spack env activate ./spack-env

# 2. Compile and install the packages listed in the environment
$ spack install

# optional?? it looks like cmake can still find the packages just by loding the environment using pkg-config
$ spack load openblas openmpi petsc
```

### CMake

To build `main` from `main.c`

```bash shell
$ rm -rf build
$ mkdir build
$ cd build
$ cmake ..
$ make
$ mpirun -n 2 ./main
```
