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
 * // Make a functor to select a device
 * class my_functor_selector {
 *   int operator()(const sycl::device& dev) const {
 *   ...
 *   }
 * }
 * ...
 * auto q = sycl::queue{my_functor_selector{}};
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * // Or use a function selector
 * int my_function_selector(const sycl::device &d) {
 *  ...
 * }
 * ...
 * auto q = sycl::queue{my_function_selector};
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * // Or use a lambda selector
 * auto my_lambda_selector = [](const sycl::device &d) {
 *  ...
 * };
 * ...
 * auto q = sycl::queue{my_lambda_selector};
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * // Query a device for some things:
 * std::string vendor = dev.get_info<sycl::info::device::vendor>();
 * std::string dev_name = dev.get_info<sycl::info::device::name>();
 * std::string dev_driver_ver =
 dev.get_info<sycl::info::device::driver_version>();
 * int bits = dev.get_info<sycl::info::device::address_bits>();
 * 
 *
*/

#include "../helpers.hpp"
#include <sycl/sycl.hpp>

int main() {
  int a = 18, b = 24, r = 0;

  try {
    // Task: add a device selector to create this queue with a GPU that has 64-bit addresses
    auto defaultQueue = sycl::queue {};

    int* A = sycl::malloc_device<int>(1, defaultQueue);
    int* B = sycl::malloc_device<int>(1, defaultQueue);
    int* R = sycl::malloc_device<int>(1, defaultQueue);

    defaultQueue.memcpy(A, &a, sizeof(int)).wait();
    defaultQueue.memcpy(B, &b, sizeof(int)).wait();

    defaultQueue.single_task([=]() { R[0] = A[0] + B[0]; }).wait();

    defaultQueue.memcpy(&r, R, sizeof(int)).wait();

    sycl::free(A, defaultQueue);
    sycl::free(B, defaultQueue);
    sycl::free(R, defaultQueue);

    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  SYCLACADEMY_ASSERT(r == 42);
}
