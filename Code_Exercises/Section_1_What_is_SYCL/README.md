# SYCL Academy

## Exercise 1: Compiling with SYCL

---

For this first exercise you simply need to install a SYCL implementation and the SYCL
Academy dependencies and then verify your installation by comping a source file
for SYCL.


### 1.) Installing a SYCL implementation

To get started for the workshop, follow these steps:
For MOGON NHR, open: mod.hpc.uni-mainz.de  
Login using the provided credentials.
Open "Code Server".
Start a new session:
- Use `ki-heprosycl` as account
- Use `A40` as partition
- Number of hours: `9`
- Number of Tasks: `1`
- CPUs per Task: `8`
- Memory: `32`
- Number of GPUs: `1`

Once the job is started: "Open VS Code".
Start with:
- CTRL+ALT+P: "Create New Terminal"
```
module load tools/Apptainer
apptainer run --nv /lustre/project/ki-heprosycl/cuda-devel.sif
```
Now you're in the apptainer environment that we'll use for today.
It provides all the dependencies we'll need today (LLVM, Boost, CMake, Ninja, git, ..)

Get and install AdaptiveCpp:
```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp
cd AdaptiveCpp
mkdir build
cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/install
ninja install
```

Now, you have AdaptiveCpp installed.


### 2.) Verifying your environment

#### When using AdaptiveCpp

With AdaptiveCpp, you can skip this step. If you suspect later that your environment might not be set up correctly, you can run `acpp-info -l` in the `bin`  directory of your AdaptiveCpp installation. It will then print the backends and devices that it sees, for example:
```
$ acpp-info -l
=================Backend information===================
Loaded backend 0: OpenCL
  Found device: Intel(R) UHD Graphics 620 [0x5917]
  Found device: ComputeAorta x86_64
  Found device: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
Loaded backend 1: OpenMP
  Found device: hipSYCL OpenMP host device
Loaded backend 2: CUDA
  Found device: NVIDIA GeForce MX150
Loaded backend 3: Level Zero
  Found device: Intel(R) UHD Graphics 620 [0x5917]
```

### 3.) Configuring the exercise project

Once you have confirmed your environment is setup and available you are ready to
compile your first SYCL application from source code.

First fetch the tutorial samples from GitHub.

Clone this repository ensuring that you include sub-modules.

```sh
git clone --recursive https://github.com/AdaptiveCpp/syclacademy.git -b nhr-uds
cd syclacademy
mkdir build
cd build
```

### 4.) Include the SYCL header file

Then open the source file for this exercise and include the SYCL header file
`"sycl/sycl.hpp"`.

Make sure before you do this you define `SYCL_LANGUAGE_VERSION` to `2020`, to
enable support for the SYCL 2020 interface.

Once that is done build your source file with your chosen build system.

### 5.) Compile and run

Once you've done that simply build the exercise with your chosen build system
and invoke the executable.

#### Build And Execution Hints

```sh
# <target specification> is a list of backends and devices to target, for example
# "generic" compiles for CPUs and GPUs using the generic single-pass compiler.
# When in doubt, use "generic" as it usually generates the fastest binaries.
#
# Recent, full installations of AdaptiveCpp may not need targets to be provided,
# compiling for "generic" by default.
# Ggf. zusätzlich: -DSYCL_ACADEMY_ENABLE_SOLUTIONS=ON
cmake -GNinja -DSYCL_ACADEMY_USE_ADAPTIVECPP=ON -DSYCL_ACADEMY_INSTALL_ROOT=~/install -DACPP_TARGETS="<target specification>" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
ninja Section_1_What_is_SYCL_source
```
alternatively, without CMake:
```sh
cd Code_Exercises/Section_1_What_is_SYCL
~/install/bin/acpp -o Section_1_What_is_SYCL_source --acpp-targets="<target specification>" source.cpp
./Section_1_What_is_SYCL_source
```

