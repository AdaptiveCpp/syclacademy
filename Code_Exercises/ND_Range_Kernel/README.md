# SYCL Academy

## Exercise 14: ND Range Kernel
---

In this exercise you will learn how to enqueue ND range kernel functions.

---

### 1.) Use items in parallel_for

Using the application from any exercise so far or creating a new one, enqueue a
kernel function using the `parallel_for` variant which takes a `range` but has
the kernel function take an `item`.
To get the item index to use to index pointers, use `itm.get_linear_id()`.

### 2.) Enqueue an ND range kernel

Using the application from any exercise so far or creating a new one, enqueue an
ND range kernel function using the `parallel_for` variant which takes an
`nd_range`.

Remember an `nd_range` is made up of two `range`s, the first being the global
range and the second being the local range or the work-group size.

Remember that when using this variant of `parallel_for` the kernel function
takes an `nd_item`.

Similarly to the `item` when using the `nd_item` you cannot pass this
directly to the subscript operator of a `pointer`, you can retrieve the linear
index by calling the `get_global_linear_id` member function.

## Build and execution hints

For DevCloud via JupiterLab follow these [instructions](../devcloudJupyter.md).

For DPC++: [instructions](../dpcpp.md).

For AdaptiveCpp: [instructions](../adaptivecpp.md).
