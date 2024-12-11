/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Construct an in-order queue
 * auto q = sycl::queue{sycl::default_selector_v,
 *        {sycl::property::queue::in_order{}}};
 *
 * // Allocate device memory
 * T* ptr = sycl::malloc_device<T>(n, q);
 * // Do a USM memcpy
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n);
 *
 * // Wait on an event
 * event.wait();
 *
 * // Wait on a queue
 * q.wait();
 *
 * // Enqueue a parallel for:
 * // i: With range:
 *        q.parallel_for<class mykernel>(sycl::range{n},
 *        [=](sycl::id<1> i) { // Do something });
 * // ii: With nd_range:
 *        q.parallel_for<class mykernel>(sycl::nd_range{
 *            globalRange, localRange}, [=](sycl::nd_item<1> i) {
 *            // Do something
 *          });
*/

#include "../helpers.hpp"

int main() {
  constexpr size_t dataSize = 1024;

  int a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = i;
    b[i] = i;
    r[i] = 0;
  }

  // Task: parallelise the vector add kernel using nd_range
  for (int i = 0; i < dataSize; ++i) {
    r[i] = a[i] + b[i];
  }

  for (int i = 0; i < dataSize; ++i) {
    SYCLACADEMY_ASSERT(r[i] == i * 2);
  }
}
