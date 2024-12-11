/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"

#include <sycl/sycl.hpp>

void test_item() {
  constexpr size_t dataSize = 1024;

  int a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = i;
    b[i] = i;
    r[i] = 0;
  }

  try {
    auto gpuQueue = sycl::queue { sycl::gpu_selector_v };

    auto A = sycl::malloc_device<int>(dataSize, gpuQueue);
    auto B = sycl::malloc_device<int>(dataSize, gpuQueue);
    auto R = sycl::malloc_device<int>(dataSize, gpuQueue);

    gpuQueue.memcpy(A, a, dataSize * sizeof(int)).wait();
    gpuQueue.memcpy(B, b, dataSize * sizeof(int)).wait();

    gpuQueue.parallel_for(
          sycl::range { dataSize }, [=](sycl::item<1> itm) {
            auto globalId = itm.get_linear_id();
            R[globalId] = A[globalId] + B[globalId];
          }).wait();

    gpuQueue.memcpy(r, R, dataSize * sizeof(int)).wait();

    gpuQueue.throw_asynchronous();
    sycl::free(A, gpuQueue);
    sycl::free(B, gpuQueue);
    sycl::free(R, gpuQueue);
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    SYCLACADEMY_ASSERT(r[i] == i * 2);
  }
}

void test_nd_item() {
  constexpr size_t dataSize = 1024;
  constexpr size_t workGroupSize = 128;

  int a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = i;
    b[i] = i;
    r[i] = 0;
  }

  try {
    auto gpuQueue = sycl::queue { sycl::gpu_selector_v };

    auto A = sycl::malloc_device<int>(dataSize, gpuQueue);
    auto B = sycl::malloc_device<int>(dataSize, gpuQueue);
    auto R = sycl::malloc_device<int>(dataSize, gpuQueue);

    gpuQueue.memcpy(A, a, dataSize * sizeof(int)).wait();
    gpuQueue.memcpy(B, b, dataSize * sizeof(int)).wait();


    auto ndRange = sycl::nd_range { sycl::range { dataSize },
                                    sycl::range { workGroupSize } };
    gpuQueue.parallel_for(
                  ndRange, [=](sycl::nd_item<1> itm) {
            auto globalId = itm.get_global_linear_id();
            R[globalId] = A[globalId] + B[globalId];
          }).wait();

    gpuQueue.memcpy(r, R, dataSize * sizeof(int)).wait();

    gpuQueue.throw_asynchronous();

    sycl::free(A, gpuQueue);
    sycl::free(B, gpuQueue);
    sycl::free(R, gpuQueue);
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    SYCLACADEMY_ASSERT(r[i] == i * 2);
  }
}

int main() {
  test_item();
  test_nd_item();
}
