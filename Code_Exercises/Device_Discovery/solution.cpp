/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"
#include <hipSYCL/sycl/queue.hpp>
#include <hipSYCL/sycl/usm.hpp>
#include <iostream>
#include <sycl/sycl.hpp>

// Function device selector
int bits64_gpu_selector1(const sycl::device& dev) {
  if (dev.has(sycl::aspect::gpu)) {
    auto bits = dev.get_info<sycl::info::device::address_bits>();
    if (bits >= 64) {
      return 1;
    }
  }
  return -1;
}

// Lambda device_selector
auto bits64_gpu_selector2 = [](const sycl::device& dev) {
  if (dev.has(sycl::aspect::gpu)) {
    auto bits = dev.get_info<sycl::info::device::address_bits>();
    if (bits >= 64) {
      return 1;
    }
  }
  return -1;
};

int main() {
  int a = 18, b = 24, r = 0;

  try {
    auto defaultQueue1 = sycl::queue { bits64_gpu_selector1 };
    auto defaultQueue2 = sycl::queue { bits64_gpu_selector2 };

    std::cout << "Chosen device: "
              << defaultQueue1.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    int* A = sycl::malloc_device<int>(1, defaultQueue1);
    int* B = sycl::malloc_device<int>(1, defaultQueue1);
    int* R = sycl::malloc_device<int>(1, defaultQueue1);

    defaultQueue1.memcpy(A, &a, sizeof(int)).wait();
    defaultQueue1.memcpy(B, &b, sizeof(int)).wait();

    defaultQueue1.single_task([=]() { R[0] = A[0] + B[0]; }).wait();

    defaultQueue1.memcpy(&r, R, sizeof(int)).wait();

    sycl::free(A, defaultQueue1);
    sycl::free(B, defaultQueue1);
    sycl::free(R, defaultQueue1);

    defaultQueue1.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
  SYCLACADEMY_ASSERT(r == 42);
}
