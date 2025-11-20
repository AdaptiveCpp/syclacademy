/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Include SYCL header
 * #include <sycl/sycl.hpp>
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Allocate device memory
 * auto * devPtr = sycl::malloc_device<int>(mycount, q);
 *
 * // Do a memcpy
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n);
 * // Do a memcpy with dependent events
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n, {event1, event2});
 *
 * // Wait on an event
 * event.wait();
 *
 * // Wait on a queue
 * q.wait();
 * 
 * // Free memory
 * sycl::free(ptr, q);
 *
 * // Submit a single task kernel
 * q.single_task([=](){
 *      // Some kernel code
 *   }).wait();
 *
*/

#include "../helpers.hpp"

int main() {
  int a = 74, b = 24, c = 18, r = 0;

  // Task: Run these kernels on the SYCL device, respecting the dependencies
  // as shown in the README

  // Kernel A:
  a = a * 2;

  // Kernel B:
  b = b + a;

  // Kernel C:
  c = c - a;

  // Kernel D:
  r = b + c;

  SYCLACADEMY_ASSERT(r == 42);
}
