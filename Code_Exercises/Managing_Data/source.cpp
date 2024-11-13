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
 * // Memcpy
 * q.memcpy(dst, src, sizeof(T)*n).wait();
 * // (dst and src are pointers)
 *
 * // Free memory
 * sycl::free(ptr, q);
 *
 * // Submit a kernel
 * q.submit([&](sycl::handler &cgh) {
 *    cgh.single_task([=](){
 *      // Some kernel code
 *      });
 * }).wait();
 *
*/

#include "../helpers.hpp"

int main() {
  int a = 18, b = 24, r = 0;

  // Task: Compute a+b on the SYCL device using USM
  r = a + b;

  SYCLACADEMY_ASSERT(r == 42);
}
