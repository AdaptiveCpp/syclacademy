# SYCL Academy

## Exercise 2: Hello World

---

In this first exercise you will learn how to enqueue your first kernel function
to run on a device and print `Hello World!` to the console.

---

### 1.) Create a queue

The first thing you must do is create a `queue` to submit work to. The simplest
way to do this is to default construct it, this will choose a device for you.

### 3.) Define a SYCL kernel function

Use the queue to submit a single task to the device.
You can use the `single_task` function, which takes only a function
object which itself doesn't take any parameters.

Remember to call `wait` on the `event` returned from `single_task` to await the
completion of the kernel function.

### 4.) Stream "Hello World!" to stdout

For now, let's use AdaptiveCpp's built-in `sycl::detail::print` in the kernel 
to print out the string `"Hello World!\n"` from the device.

#### Build And Execution Hints

For AdaptiveCpp: [instructions](../adaptivecpp.md).
