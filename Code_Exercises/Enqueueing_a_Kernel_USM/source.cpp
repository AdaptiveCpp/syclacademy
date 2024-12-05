/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 *
 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Include SYCL header
 * #include <sycl/sycl.hpp>
 *
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Submit a single job to the queue
 * q.single_task([=]() {
 *   // do something
 * });
 *
 * // Write on the stdout from the device (AdaptiveCpp only - non-portable!)
 * sycl::detail::print("Hello World!\n");
 * 
 */

#include "../helpers.hpp"
#include <iostream>

int main() {

  // Print "Hello World!\n"
  std::cout << "Hello World!\n";

  // Task: Have this message print from the SYCL device instead of the host

  SYCLACADEMY_ASSERT(true);
}
