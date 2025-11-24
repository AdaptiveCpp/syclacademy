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
  int a = 74, b = 24, c = 18, r = 0;

  auto defaultQueue = sycl::queue{};

  auto dev_A = sycl::malloc_device<int>(1, defaultQueue);
  auto dev_B = sycl::malloc_device<int>(1, defaultQueue);
  auto dev_C = sycl::malloc_device<int>(1, defaultQueue);
  auto dev_R = sycl::malloc_device<int>(1, defaultQueue);

  auto eDA = defaultQueue.memcpy(dev_A, &a, 1 * sizeof(int));
  auto eDB = defaultQueue.memcpy(dev_B, &b, 1 * sizeof(int));
  auto eDC = defaultQueue.memcpy(dev_C, &c, 1 * sizeof(int));

  // Kernel A:
  auto eKA = defaultQueue.single_task(eDA,
      [=] {
        *dev_A = *dev_A * 2;
      }
    );

  // Kernel B:
  auto eKB = defaultQueue.single_task({eKA, eDB},
      [=] {
        *dev_B = *dev_B + *dev_A;
      }
    );

  // Kernel C:
  auto eKC = defaultQueue.single_task({eKA, eDC},
      [=] {
        *dev_C = *dev_C - *dev_A;
      }
    );

  // Kernel D:
  auto eKD = defaultQueue.single_task({eKC},
      [=] {
        *dev_R = *dev_B + *dev_C;
      }
    );

  defaultQueue.memcpy(&r, dev_R, 1 * sizeof(int), eKD).wait();

  sycl::free(dev_A, defaultQueue);
  sycl::free(dev_B, defaultQueue);
  sycl::free(dev_C, defaultQueue);
  sycl::free(dev_R, defaultQueue);

  SYCLACADEMY_ASSERT(r == 42);
}
