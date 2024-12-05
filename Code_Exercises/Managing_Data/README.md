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

### 4.) Declare your kernel

Declare a SYCL kernel function using the `single_task` command and providing a
lambda as the kernel function. The kernel function can just dereference the device
pointer with `*p` or `p[]` to read from the inputs and write the sum to the output.
As you want to dereference the first value pointed to by `p`, you can just use `*p`
or `p[0]`.

### 5.) Move your data back

After the kernel is done executing, you have to copy the result back to the host,
before you can use the value.

### 6.) Don't forget to synchronize!

The USM memory mangement model requires you to explicitly synchronize between
asynchronous operations (memory movement, kernels, ..).
For now, use `wait()` on all asynchronous operations, we will see how to do this
better, in Data and Dependencies.


#### Build And Execution Hints

For DevCloud via JupiterLab follow these [instructions](../devcloudJupyter.md).

For DPC++: [instructions](../dpcpp.md).


For AdaptiveCpp: [instructions](../adaptivecpp.md).
