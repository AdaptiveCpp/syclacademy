# SYCL Academy

## Exercise 3: Scalar Add

---

In this exercise you will learn how to allocate device memory, move data to
the device and how to access the data within a kernel function.

---

### 1.) Allocate your input and output

Allocate memory on the host for your input and output data variables and assign
values to the inputs.

### 2.) Allocate device memory

Use `sycl::malloc_device` to allocate a sufficiently large memory region on the device.

### 3.) Move your host data to the device

Your input values still are only available on the host.
To make them available on the device as well, use `queue::memcpy` to copy the
local values to the allocated device memory.

### 4.) Declare your kernels

Declare SYCL kernel functions using the `single_task` command and providing a
lambda as the kernel functions. The kernel function can just dereference the device
pointer with `*p` or `p[]` to read from the inputs and write the result to the target pointer.
As you want to dereference the first value pointed to by `p`, you can just use `*p`

### 5.) Define a data flow graph with the USM model

Now define the kernel dependencies, so that they are executed in the correct order.
These kernel functions can do any computation you like, but they should follow the following dependencies.

          (kernel A)
         /          \
    (kernel B)  (kernel C)
         \          /
          (kernel D)

The important thing here is that kernels B and C must depend on kernel A, kernel
D must depend on kernels B and C and kernels B and C can be executed in any
order and even concurrently if the device permits. Note that in the USM model
dependencies are defined explicitly by chaining commands via `event`s.

Feel free to use any method of synchronization and copy back you like.

### 5.) Move your data back

After the kernels are done executing, you have to copy the result back to the host,
before you can use the value.

### 6.) Don't forget to synchronize!

The USM memory management model requires you to explicitly synchronize between
asynchronous operations (memory movement, kernels, ..).
Don't forget to synchronize after the final memcpy to make sure the data is available
before you read it on the host.


#### Build And Execution Hints

For DevCloud via JupiterLab follow these [instructions](../devcloudJupyter.md).

For DPC++: [instructions](../dpcpp.md).


For AdaptiveCpp: [instructions](../adaptivecpp.md).
