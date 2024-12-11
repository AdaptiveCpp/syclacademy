/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"

#include <sycl/sycl.hpp>

int main() {
  int a = 18, b = 24, r = 0;

  auto defaultQueue = sycl::queue {};

  auto dev_A = sycl::malloc_device<int>(1, defaultQueue);
  auto dev_B = sycl::malloc_device<int>(1, defaultQueue);
  auto dev_R = sycl::malloc_device<int>(1, defaultQueue);

  defaultQueue.memcpy(dev_A, &a, 1 * sizeof(int)).wait();
  defaultQueue.memcpy(dev_B, &b, 1 * sizeof(int)).wait();

  defaultQueue.single_task(
      [=] {
        dev_R[0] = dev_A[0] + dev_B[0];
      }
    ).wait();

  defaultQueue.memcpy(&r, dev_R, 1 * sizeof(int)).wait();

  sycl::free(dev_A, defaultQueue);
  sycl::free(dev_B, defaultQueue);
  sycl::free(dev_R, defaultQueue);

  SYCLACADEMY_ASSERT(r == 42);
}
