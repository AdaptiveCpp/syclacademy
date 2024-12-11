# SYCL Academy

## Exercise 10: Managing Dependencies
---

In this exercise you will learn how to create a data dependency data flow graph.

---

### 1.) Define a data flow graph with the USM model

Using everything you have learned in previous exercises create an application
which has four kernel functions. These kernel functions can do any computation
you like, but they should follow the following dependencies.

          (kernel A)
         /          \
    (kernel B)  (kernel C)
         \          /
          (kernel D)

The important thing here is that kernels B and C must depend on kernel A, kernel
D must depend on kernels B and C and kernels B and C can be executed in any
order and even concurrently if the device permits. Note that in the USM model
dependencies are defined explicitly by chaining commands via `event`s.

Feel free to use any method of synchronization and copy back you like,
but remember to handle errors.

#### Build And Execution Hints

For DevCloud via JupiterLab follow these [instructions](../devcloudJupyter.md).

For DPC++: [instructions](../dpcpp.md).

For AdaptiveCpp: [instructions](../adaptivecpp.md).